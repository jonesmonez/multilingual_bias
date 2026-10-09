from datetime import datetime

def filename(
    method,
    bias_type=None,
    lang_debias=None,
    lang_eval=None,
    model_name_or_path=None,
    time: bool = False,
):
    if time:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_"
    else:
        timestamp = ""
    
    experiment_id = timestamp

    if isinstance(method, str):
        experiment_id += f"{method}"
    if isinstance(bias_type, str):
        experiment_id += f"_{bias_type}"
    if isinstance(lang_debias, str):
        experiment_id += f"_{lang_debias}"
    if isinstance(lang_eval, str):
        experiment_id += f"_eval-{lang_eval}"
    if isinstance(model_name_or_path, str):
        experiment_id += f"_{model_name_or_path}"

    return experiment_id
