import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
    
families = {
    "romanesque": ["fr_FR", "es_AR", "it_IT", "ca_ES"],
    "germanic": ["de_DE", "en_US"],
    "semitic": ["ar_DZ", "mt_MT"],
    "sinotibetan": ["zh_CN"],
}

methods = ["cda", "dropout", "inlp", "sentdebias", "densray"]
btypes = ["gender", "race-color", "religion"]

def get_family(lang_code):
    for family, langs in families.items():
        if lang_code in langs:
            return family
    return "unk"

def classify_lang_pair(dlang, elang):
    if dlang == elang:
        return "mono"
    
    if "mt_MT" in (dlang, elang):
        return "special"
    
    dlang_fam = get_family(dlang)
    elang_fam = get_family(elang)
    
    if "unk" in (dlang_fam, elang_fam):
        raise ValueError(f"Unknown Language: {dlang}, {elang}")
    
    if dlang_fam == elang_fam:
        if dlang_fam == "romanesque":
            return "intra"
        return "excluded"

    return "inter"

def transform_json(json_path: str, base_path: str):
    with open(json_path, "r") as f:
        input_json = json.load(f)
        
    with open(base_path, "r") as bf:
        input_base = json.load(bf)
    
    return_list = []
        
    for method, mvalues in input_json.items():
        for dlang, dvalues in mvalues.items():
            for elang, evalues in dvalues.items():
                for btype, bvalue in evalues.items():
                    base_value = input_base.get(elang, {}).get(btype, None)
                    
                    if base_value is None or bvalue is None:
                        continue
                    
                    base_diff = abs(base_value - 50)
                    deb_diff = abs(bvalue - 50)
                    delta = base_diff - deb_diff
                    
                    ri = (delta / base_diff) * 100
                    
                    datapoint = {
                        "dlang": dlang,
                        "elang": elang,
                        "delta": round(delta, 2),
                        "ri": round(ri, 2),
                        "method": method,
                        "bias_type": btype,
                        "baseline_score": base_value,
                        "baseline_diff": round(base_diff, 2),
                        "debiased_score": bvalue,
                        "debiased_diff": round(deb_diff, 2),
                        "family_dlang": get_family(dlang),
                        "family_elang": get_family(elang),
                        "relationship": classify_lang_pair(dlang, elang),
                    }
                    # print(datapoint)
                    return_list.append(datapoint)
                    
    return return_list

class DebiasStatistics:
    def __init__(
        self,
        data: list | None = None,
        save_path: str = "statresults"
    ):
        if data is None:
            self.data = transform_json("results6.json", "results_base.json")
        else:
            self.data = data
            
        self.df = pd.DataFrame(self.data)
        
        self.save_path = save_path
    
    def load_data(
        self,
        json_path: str,
        base_path: str,
    ):
        self.data = transform_json(json_path, base_path)
        
    def save_data_as_csv(
        self,
    ):
        self.df.to_csv("data.csv")

    def mono_baseline(
        self,
        save_result: bool = False
    ):
        results = {}
        df = pd.DataFrame(self.data)
        df = df[df["dlang"] == df["elang"]]
        for btype in btypes:
            btype_df = df[df["bias_type"] == btype]
            pivot_df = btype_df.pivot(
                index="elang",
                columns="method",
                values="delta"
            )
            pivot_df["mean"] = pivot_df.mean(axis=1, numeric_only=True)
            pivot_df.loc["mean"] = pivot_df.mean(numeric_only=True)

            results[btype] = pivot_df.round(2)

        for btype, result in results.items():
            print(f"{btype}:")
            print(result.to_string())
            print()
            if save_result:
                os.makedirs(self.save_path, exist_ok=True)
                results_df.to_csv(f"{self.save_path}/mono_baseline_{btype}.csv", index=False, encoding="utf-8")
        
    def significance(
        self,
        save_result: bool = False
    ):
        df = pd.DataFrame(self.data)
        df = df[df["dlang"] != df["elang"]]
        
        raw_results = []
        
        for method in methods:
            for btype in btypes:
                df_test = df[(df["method"] == method) & (df["bias_type"] == btype)]
                
                if len(df_test) < 5:
                    continue
                
                delta_intra = df_test[df_test["relationship"] == "intra"]["delta"].dropna().values
                delta_inter = df_test[df_test["relationship"] == "inter"]["delta"].dropna().values
                
                if len(delta_intra) < 3 or len(delta_inter) < 3:
                    continue
                
                _, p_intra = stats.shapiro(delta_intra)
                _, p_inter = stats.shapiro(delta_inter)
                
                n1, n2 = len(delta_intra), len(delta_inter)
                mean_intra, mean_inter = np.mean(delta_intra), np.mean(delta_inter)
                
                if p_intra > 0.05 and p_inter > 0.05:
                    _, p_val = stats.ttest_ind(delta_intra, delta_inter, equal_var=False, alternative="greater")
                    _, p_two = stats.ttest_ind(delta_intra, delta_inter, equal_var=False, alternative="two-sided")
                    test_used = "t-test"
                    
                    s1, s2 = np.std(delta_intra, ddof=1), np.std(delta_inter, ddof=1)
                    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                    effect_size = (mean_intra - mean_inter) / s_pooled if s_pooled > 0 else 0
                else:
                    u_stat, p_val = stats.mannwhitneyu(delta_intra, delta_inter, alternative="greater")
                    _, p_two = stats.mannwhitneyu(delta_intra, delta_inter, alternative="two-sided")
                    test_used = "mann-whitney"
                    
                    effect_size = (2 * u_stat) / (n1 * n2) - 1
                                
                raw_results.append({
                    "method": method,
                    "bias_type": btype,
                    "n_intra": n1,
                    "n_inter": n2,
                    "mean_intra": round(mean_intra, 2),
                    "mean_inter": round(mean_inter, 2),
                    "p_value": round(p_val, 4),
                    "p_value_two": round(p_two, 4),
                    "p_intra": round(p_intra, 2),
                    "p_inter": round(p_inter, 2),
                    "effect_size": round(effect_size, 2),
                    "supports_h1": mean_intra > mean_inter,
                    "test_used": test_used
                })

        tests_performed = len(raw_results)
        alpha_adjusted = 0.05 / tests_performed if tests_performed > 0 else 0.05
        
        for res in raw_results:
            res["significant"] = res["p_value"] < alpha_adjusted
            res["sig_direction"] = (
                "supports_H1" if (res["significant"] and res["supports_h1"])
                else "against_H1" if res["significant"]
                else "not significant"
            )

        results_df = pd.DataFrame(raw_results)
        print("\n=== STATISTICAL ANALYSIS RESULTS ===")
        print(results_df.to_string())
        print(f"\nBonferroni-adjusted alpha: {alpha_adjusted:.4f}")

        if save_result:
            os.makedirs(self.save_path, exist_ok=True)
            results_df.to_csv(f"{self.save_path}/significance.csv", index=False, encoding="utf-8")       

    def boxplot(
        self,
        btype: str,
        save_result: bool = False
    ):
        df = pd.DataFrame(self.data)
        df = df[(df["bias_type"] == btype) & (df["dlang"] != df["elang"])]

        methods_present = [m for m in methods if df[df["method"] == m]["relationship"].isin(["intra", "inter"]).any()]

        fig, axes = plt.subplots(
            len(methods_present), 1,
            figsize=(9, 2.2 * len(methods_present)),
            sharex=True
        )
        if len(methods_present) == 1:
            axes = [axes]

        for ax, method in zip(axes, methods_present):
            df_m = df[df["method"] == method]
            groups = [
                df_m.loc[df_m["relationship"] == "inter", "delta"]
                    .dropna()
                    .to_numpy(dtype=float),
                df_m.loc[df_m["relationship"] == "intra", "delta"]
                    .dropna()
                    .to_numpy(dtype=float),
            ]

            bp = ax.boxplot(
                groups,
                orientation="horizontal",
                tick_labels=["inter", "intra"],
                widths=0.5,
                patch_artist=True,
                showmeans=True,
                showfliers=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "#ffb300",
                    "markeredgecolor": "#ffb300",
                    "markersize": 3,
                },
                medianprops={"color": "black"},
            )

            for patch, color in zip(
                bp["boxes"],
                ["#9e9e9e", "#6d4aff"]
            ):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            rng = np.random.default_rng(42)

            for i, g in enumerate(groups, start=1):
                if len(g) == 0:
                    continue
                
                q1, q3 = np.percentile(g, [25, 75])
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                is_outlier = (g < lower) | (g > upper)
                normal = g[~is_outlier]
                outliers = g[is_outlier]
                y_normal = rng.normal(i, 0.05, size=len(normal))

                ax.scatter(
                    normal,
                    y_normal,
                    s=12,
                    color="#333333",
                    alpha=0.5,
                    zorder=3,
                )

                y_outlier = np.full(
                    len(outliers),
                    i,
                    dtype=float
                )

                ax.scatter(
                    outliers,
                    y_outlier,
                    s=12,
                    facecolors="white",
                    edgecolors="black",
                    marker="o",
                )

            ax.axvline(0, color="red", linestyle=":", linewidth=1, alpha=0.7)
            ax.set_title(method, fontsize=10, fontweight="bold", loc="left")
            ax.grid(True, axis="x", alpha=0.3, linestyle=":")

        axes[-1].set_xlabel(r"$\Delta$ (Prozentpunkte)")
        fig.suptitle(f"Transfer-Gain nach Familiengruppe — Bias-Kategorie: {btype}",
                    fontsize=13, fontweight="bold")
        fig.tight_layout()

        if save_result:
            os.makedirs(self.save_path, exist_ok=True)
            path = os.path.join(self.save_path, f"boxplot_{btype}.pdf")
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved: {path}")
            plt.close(fig)
        else:
            plt.show()

    def boxplot_stats(
        self,
        btype: str = "gender",
        save_result: bool = False
    ):
        df = pd.DataFrame(self.data)
        df = df[(df["bias_type"] == btype) & (df["dlang"] != df["elang"])]

        rows = []
        for method in methods:
            df_m = df[df["method"] == method]
            for rel in ["intra", "inter"]:
                vals = df_m[df_m["relationship"] == rel]["delta"].dropna()
                if len(vals) < 3:
                    continue

                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                rows.append({
                    "method": method,
                    "group": rel,
                    "n": len(vals),
                    "median": round(med, 2),
                    "mean": round(np.mean(vals), 2),
                    "std": round(np.std(vals, ddof=1), 2),
                    "q1": round(q1, 2),
                    "q3": round(q3, 2),
                    "iqr": round(q3 - q1, 2),
                    "min": round(vals.min(), 2),
                    "max": round(vals.max(), 2),
                    "skew": round(stats.skew(vals, bias=False), 2),
                    "n_outliers": int(
                        ((vals < q1 - 1.5 * (q3 - q1)) | (vals > q3 + 1.5 * (q3 - q1))).sum()
                    ),
                })

        result = pd.DataFrame(rows)
        print(f"\n=== DESCRIPTIVE STATISTICS (bias_type: {btype}) ===")
        print(result.to_string(index=False))

        if save_result:
            os.makedirs(self.save_path, exist_ok=True)
            result.to_csv(f"boxplot_stats_{btype}.csv", index=False, encoding="utf-8")

        return result

    def intra_transfer_results(
        self,
        fam: str = "romanesque",
        btype: str = "gender",
        save_result: bool = False
    ):
        df = self.df[(self.df["family_dlang"] == fam) & (self.df["family_elang"] == fam) & (self.df["elang"] != self.df["dlang"]) & (self.df["bias_type"] == btype)]

        df = df.pivot_table(
            index=["elang", "dlang"],
            columns="method",
            values="delta",
        )

        df["mean"] = df.mean(axis=1, numeric_only=True)

        df.loc[("mean", "all"), :] = df.mean(numeric_only=True)

        print(df.round(2))
        
        if save_result:
            os.makedirs(self.save_path, exist_ok=True)
            fd.to_csv(f"intra_transfer_{btype}.csv", index=False, encoding="utf-8")

    
    def analyze_maltese(
        self,
        save_result: bool = False,
        low_baseline_threshold: float = 1.0
    ):
        df = pd.DataFrame(self.data).copy()

        method_order = [
            "cda", "dropout", "sentdebias", "inlp", "densray"
        ]
        method_names = {
            "cda": "CDA",
            "dropout": "DO",
            "sentdebias": "SentDebias",
            "inlp": "INLP",
            "densray": "DensRay"
        }
        bias_order = ["gender", "race-color", "religion"]
        group_order = ["Romanisch (4)", "Semitisch (1)"]

        src = df[
            (df["dlang"] == "mt_MT") &
            (df["elang"] != "mt_MT")
        ].copy()

        source_summary = (
            src.groupby(["method", "bias_type"])
            .agg(
                n_pairs=("delta", "count"),
                mean_delta=("delta", "mean"),
                median_delta=("delta", "median"),
                mean_ri=("ri", "mean"),
                median_ri=("ri", "median")
            )
            .round(2)
            .reset_index()
        )

        source_summary["method"] = source_summary["method"].map(
            method_names
        )

        source_summary["method"] = pd.Categorical(
            source_summary["method"],
            categories=list(method_names.values()),
            ordered=True
        )
        source_summary["bias_type"] = pd.Categorical(
            source_summary["bias_type"],
            categories=bias_order,
            ordered=True
        )

        source_summary = source_summary.sort_values(
            ["method", "bias_type"]
        ).reset_index(drop=True)

        print("\n=== MALTESE AS SOURCE ===")
        print(source_summary.to_string(index=False))

        romance_langs = [
            "fr_FR", "es_AR", "it_IT", "ca_ES"
        ]
        selected_sources = romance_langs + ["ar_DZ"]

        tgt = df[
            (df["elang"] == "mt_MT") &
            (df["dlang"].isin(selected_sources))
        ].copy()

        tgt["src_group"] = np.where(
            tgt["dlang"] == "ar_DZ",
            "Semitisch (1)",
            "Romanisch (4)"
        )

        tgt["low_baseline"] = (
            tgt["baseline_diff"] < low_baseline_threshold
        )

        target_summary = (
            tgt.groupby(["bias_type", "src_group", "method"])
            .agg(
                n_pairs=("delta", "count"),
                mean_delta=("delta", "mean"),
                mean_ri=("ri", "mean")
            )
            .reset_index()
        )

        target_delta = target_summary.pivot(
            index=["bias_type", "src_group"],
            columns="method",
            values="mean_delta"
        )

        target_ri = target_summary.pivot(
            index=["bias_type", "src_group"],
            columns="method",
            values="mean_ri"
        )

        row_index = pd.MultiIndex.from_product(
            [bias_order, group_order],
            names=["bias_type", "src_group"]
        )

        def format_target_table(table):
            return (
                table
                .reindex(index=row_index, columns=method_order)
                .rename(columns=method_names)
                .round(2)
            )

        target_delta = format_target_table(target_delta)
        target_ri = format_target_table(target_ri)

        print("\n=== MALTESE AS EVAL (DELTA) ===")
        print(target_delta.to_string())

        print("\n=== MALTESE AS EVAL (RI) ===")
        print(target_ri.to_string())

        target_details = tgt[[
            "method",
            "bias_type",
            "dlang",
            "elang",
            "src_group",
            "baseline_diff",
            "delta",
            "ri",
            "low_baseline"
        ]].sort_values(
            ["bias_type", "method", "src_group", "dlang"]
        ).reset_index(drop=True)

        valid = target_details[
            (target_details["baseline_diff"] > 0)
            & target_details["delta"].notna()
            & target_details["ri"].notna()
        ].copy()

        valid["ri_expected"] = (
            valid["delta"] / valid["baseline_diff"] * 100
        )
        valid["ri_error"] = (
            valid["ri"] - valid["ri_expected"]
        ).abs()

        suspicious = valid[valid["ri_error"] > 0.5]

        if not suspicious.empty:
            print("\Warning: Delta, RI are not consistent")
            print(suspicious[[
                "method", "bias_type", "dlang",
                "delta", "ri", "ri_expected", "ri_error"
            ]].round(2).to_string(index=False))

        if save_result:
            os.makedirs(self.save_path, exist_ok=True)
            source_summary.to_csv(
                f"{self.save_path}/maltese_source_table_5_6.csv",
                index=False
            )
            target_delta.to_csv(
                f"{self.save_path}/maltese_target_delta_table_5_8.csv"
            )
            target_ri.to_csv(
                f"{self.save_path}/maltese_target_ri_table_A_6.csv"
            )
            target_details.to_csv(
                f"{self.save_path}/maltese_target_details.csv",
                index=False
            )
        return {
            "source": source_summary,
            "target_delta": target_delta,
            "target_ri": target_ri,
            "target_details": target_details
        }

                    
    def analyze_resource(
        self,
        langs=("mt_MT", "en_US"),
        save_result: bool = False
    ):
        df = pd.DataFrame(self.data)
        for lang in langs:
            sub = df[(df["dlang"] == lang) & (df["elang"] != lang)]
            summary = sub.groupby(["method", "bias_type"])["delta"].agg(
                ["mean", "median", "std", "count"]
            ).round(2)
            print(f"\n=== {lang} AS SOURCE ===")
            print(summary.to_string())
            
            if save_result:
                os.makedirs(self.save_path, exist_ok=True)
                summary.to_csv(f"{self.save_path}/{lang}_as_dlang.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    st = DebiasStatistics()

    st.mono_baseline()
    st.intra_transfer_results("romanesque", "gender")
    st.boxplot("gender", ".")
    st.boxplot_stats("gender", ".")
    st.significance()
    st.analyze_resource()
    st.analyse_maltese()