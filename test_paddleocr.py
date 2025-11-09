import abc
import warnings

import yaml

from paddlex import create_pipeline, create_predictor
from paddlex.inference import PaddlePredictorOption, load_pipeline_config
from paddlex.utils.config import AttrDict
from paddlex.utils.deps import DependencyError
from paddlex.utils.device import get_default_device, parse_device

_SUPPORTED_OCR_VERSIONS = ["PP-OCRv3", "PP-OCRv4", "PP-OCRv5"]

DEFAULT_DEVICE = None
DEFAULT_USE_TENSORRT = False
DEFAULT_PRECISION = "fp32"
DEFAULT_ENABLE_MKLDNN = True
DEFAULT_MKLDNN_CACHE_CAPACITY = 10
DEFAULT_CPU_THREADS = 10
SUPPORTED_PRECISION_LIST = ["fp32", "fp16"]
DEFAULT_USE_CINN = False


def parse_common_args(kwargs, *, default_enable_hpi):
    default_vals = {
        "device": DEFAULT_DEVICE,
        "enable_hpi": default_enable_hpi,
        "use_tensorrt": DEFAULT_USE_TENSORRT,
        "precision": DEFAULT_PRECISION,
        "enable_mkldnn": DEFAULT_ENABLE_MKLDNN,
        "mkldnn_cache_capacity": DEFAULT_MKLDNN_CACHE_CAPACITY,
        "cpu_threads": DEFAULT_CPU_THREADS,
        "enable_cinn": DEFAULT_USE_CINN,
    }

    unknown_names = kwargs.keys() - default_vals.keys()
    for name in unknown_names:
        raise ValueError(f"Unknown argument: {name}")

    kwargs = {**default_vals, **kwargs}

    if kwargs["precision"] not in SUPPORTED_PRECISION_LIST:
        raise ValueError(f"Invalid precision: {kwargs['precision']}. Supported values are: {SUPPORTED_PRECISION_LIST}.")

    kwargs["use_pptrt"] = kwargs.pop("use_tensorrt")
    kwargs["pptrt_precision"] = kwargs.pop("precision")
    return kwargs


def prepare_common_init_args(model_name, common_args):
    device = common_args["device"]
    if device is None:
        device = get_default_device()
    device_type, _ = parse_device(device)

    init_kwargs = {}
    init_kwargs["device"] = device
    init_kwargs["use_hpip"] = common_args["enable_hpi"]

    pp_option = PaddlePredictorOption()
    if device_type == "gpu":
        if common_args["use_pptrt"]:
            if common_args["pptrt_precision"] == "fp32":
                pp_option.run_mode = "trt_fp32"
            else:
                assert common_args["pptrt_precision"] == "fp16", common_args["pptrt_precision"]
                pp_option.run_mode = "trt_fp16"
        else:
            pp_option.run_mode = "paddle"
    elif device_type == "cpu":
        enable_mkldnn = common_args["enable_mkldnn"]
        if enable_mkldnn:
            pp_option.mkldnn_cache_capacity = common_args["mkldnn_cache_capacity"]
        else:
            pp_option.run_mode = "paddle"
        pp_option.cpu_threads = common_args["cpu_threads"]
    else:
        pp_option.run_mode = "paddle"
    pp_option.enable_cinn = common_args["enable_cinn"]
    init_kwargs["pp_option"] = pp_option

    return init_kwargs


class PaddleXPredictorWrapper(metaclass=abc.ABCMeta):
    def __init__(self, *, model_name=None, model_dir=None, **common_args):
        super().__init__()
        self._model_name = (model_name if model_name is not None else self.default_model_name)
        self._model_dir = model_dir
        self._common_args = parse_common_args(common_args, default_enable_hpi=False)
        self.paddlex_predictor = self._create_paddlex_predictor()

    @property
    @abc.abstractmethod
    def default_model_name(self):
        raise NotImplementedError

    def predict_iter(self, *args, **kwargs):
        return self.paddlex_predictor.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        result = list(self.predict_iter(*args, **kwargs))
        return result

    def close(self):
        self.paddlex_predictor.close()

    @classmethod
    @abc.abstractmethod
    def get_cli_subcommand_executor(cls):
        raise NotImplementedError

    def _get_extra_paddlex_predictor_init_args(self):
        return {}

    def _create_paddlex_predictor(self):
        kwargs = prepare_common_init_args(self._model_name, self._common_args)
        kwargs = {**self._get_extra_paddlex_predictor_init_args(), **kwargs}
        # Should we check model names?
        try:
            return create_predictor(model_name=self._model_name, model_dir=self._model_dir, **kwargs)
        except DependencyError as e:
            raise RuntimeError("A dependency error occurred during predictor creation. Please refer to the "
                               "installation documentation to ensure all required dependencies are installed.") from e


class TextDetectionMixin:
    def __init__(self, *, limit_side_len=None, limit_type=None, thresh=None, box_thresh=None, unclip_ratio=None,
                 input_shape=None, **kwargs):
        self._extra_init_args = {
            "limit_side_len": limit_side_len,
            "limit_type": limit_type,
            "thresh": thresh,
            "box_thresh": box_thresh,
            "unclip_ratio": unclip_ratio,
            "input_shape": input_shape,
        }
        super().__init__(**kwargs)

    def _get_extra_paddlex_predictor_init_args(self):
        return self._extra_init_args


class TextDetection(TextDetectionMixin, PaddleXPredictorWrapper):
    @property
    def default_model_name(self):
        return "PP-OCRv5_server_det"

    @classmethod
    def get_cli_subcommand_executor(cls):
        return


class TextRecognition(PaddleXPredictorWrapper):
    def __init__(self, *, input_shape=None, **kwargs):
        self._extra_init_args = {"input_shape": input_shape}
        super().__init__(**kwargs)

    @property
    def default_model_name(self):
        return "PP-OCRv5_server_rec"

    @classmethod
    def get_cli_subcommand_executor(cls):
        return

    def _get_extra_paddlex_predictor_init_args(self):
        return self._extra_init_args


def _merge_dicts(d1, d2):
    res = d1.copy()
    for k, v in d2.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = _merge_dicts(res[k], v)
        else:
            res[k] = v
    return res


def _to_builtin(obj):
    if isinstance(obj, AttrDict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_builtin(item) for item in obj]
    else:
        return obj


class PaddleXPipelineWrapper(metaclass=abc.ABCMeta):
    def __init__(self, *, paddlex_config=None, **common_args):
        super().__init__()
        self._paddlex_config = paddlex_config
        self._common_args = parse_common_args(common_args, default_enable_hpi=None)
        self._merged_paddlex_config = self._get_merged_paddlex_config()
        self.paddlex_pipeline = self._create_paddlex_pipeline()

    @property
    @abc.abstractmethod
    def _paddlex_pipeline_name(self):
        raise NotImplementedError

    def export_paddlex_config_to_yaml(self, yaml_path):
        with open(yaml_path, "w", encoding="utf-8") as f:
            config = _to_builtin(self._merged_paddlex_config)
            yaml.safe_dump(config, f)

    def close(self):
        self.paddlex_pipeline.close()

    @classmethod
    @abc.abstractmethod
    def get_cli_subcommand_executor(cls):
        raise NotImplementedError

    def _get_paddlex_config_overrides(self):
        return {}

    def _get_merged_paddlex_config(self):
        if self._paddlex_config is None:
            config = load_pipeline_config(self._paddlex_pipeline_name)
        elif isinstance(self._paddlex_config, str):
            config = load_pipeline_config(self._paddlex_config)
        else:
            config = self._paddlex_config

        overrides = self._get_paddlex_config_overrides()

        return _merge_dicts(config, overrides)

    def _create_paddlex_pipeline(self):
        kwargs = prepare_common_init_args(None, self._common_args)
        try:
            return create_pipeline(config=self._merged_paddlex_config, **kwargs)
        except DependencyError as e:
            raise RuntimeError("A dependency error occurred during pipeline creation. Please refer to the installation "
                               "documentation to ensure all required dependencies are installed.") from e


def create_config_from_structure(structure, *, unset=None, config=None):
    if config is None:
        config = {}
    for k, v in structure.items():
        if v is unset:
            continue
        idx = k.find(".")
        if idx == -1:
            config[k] = v
        else:
            sk = k[:idx]
            if sk not in config:
                config[sk] = {}
            create_config_from_structure({k[idx + 1:]: v}, config=config[sk])
    return config


class PaddleOCR(PaddleXPipelineWrapper):
    def __init__(
            self,
            doc_orientation_classify_model_name=None,
            doc_orientation_classify_model_dir=None,
            doc_unwarping_model_name=None,
            doc_unwarping_model_dir=None,
            text_detection_model_name=None,
            text_detection_model_dir=None,
            textline_orientation_model_name=None,
            textline_orientation_model_dir=None,
            textline_orientation_batch_size=None,
            text_recognition_model_name=None,
            text_recognition_model_dir=None,
            text_recognition_batch_size=None,
            use_doc_orientation_classify=None,
            use_doc_unwarping=None,
            use_textline_orientation=None,
            text_det_limit_side_len=None,
            text_det_limit_type=None,
            text_det_thresh=None,
            text_det_box_thresh=None,
            text_det_unclip_ratio=None,
            text_det_input_shape=None,
            text_rec_score_thresh=None,
            return_word_box=None,
            text_rec_input_shape=None,
            lang=None,
            ocr_version=None,
            **kwargs,
    ):
        if ocr_version is not None and ocr_version not in _SUPPORTED_OCR_VERSIONS:
            raise ValueError(f"Invalid OCR version: {ocr_version}. Supported values are {_SUPPORTED_OCR_VERSIONS}.")

        if all(
                map(
                    lambda p: p is None,
                    (
                            text_detection_model_name,
                            text_detection_model_dir,
                            text_recognition_model_name,
                            text_recognition_model_dir,
                    ),
                )
        ):
            if lang is not None or ocr_version is not None:
                det_model_name, rec_model_name = self._get_ocr_model_names(lang, ocr_version)
                if det_model_name is None or rec_model_name is None:
                    raise ValueError(f"No models are available for the language {repr(lang)} and "
                                     f"OCR version {repr(ocr_version)}.")
                text_detection_model_name = det_model_name
                text_recognition_model_name = rec_model_name
        else:
            if lang is not None or ocr_version is not None:
                warnings.warn("`lang` and `ocr_version` will be ignored when model names or "
                              "model directories are not `None`.", stacklevel=2)

        params = {
            "doc_orientation_classify_model_name": doc_orientation_classify_model_name,
            "doc_orientation_classify_model_dir": doc_orientation_classify_model_dir,
            "doc_unwarping_model_name": doc_unwarping_model_name,
            "doc_unwarping_model_dir": doc_unwarping_model_dir,
            "text_detection_model_name": text_detection_model_name,
            "text_detection_model_dir": text_detection_model_dir,
            "textline_orientation_model_name": textline_orientation_model_name,
            "textline_orientation_model_dir": textline_orientation_model_dir,
            "textline_orientation_batch_size": textline_orientation_batch_size,
            "text_recognition_model_name": text_recognition_model_name,
            "text_recognition_model_dir": text_recognition_model_dir,
            "text_recognition_batch_size": text_recognition_batch_size,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
            "text_det_limit_side_len": text_det_limit_side_len,
            "text_det_limit_type": text_det_limit_type,
            "text_det_thresh": text_det_thresh,
            "text_det_box_thresh": text_det_box_thresh,
            "text_det_unclip_ratio": text_det_unclip_ratio,
            "text_det_input_shape": text_det_input_shape,
            "text_rec_score_thresh": text_rec_score_thresh,
            "return_word_box": return_word_box,
            "text_rec_input_shape": text_rec_input_shape,
        }
        base_params = {}

        self._params = params

        super().__init__(**base_params)

    @property
    def _paddlex_pipeline_name(self):
        return "OCR"

    def predict_iter(
            self,
            input,
            *,
            use_doc_orientation_classify=None,
            use_doc_unwarping=None,
            use_textline_orientation=None,
            text_det_limit_side_len=None,
            text_det_limit_type=None,
            text_det_thresh=None,
            text_det_box_thresh=None,
            text_det_unclip_ratio=None,
            text_rec_score_thresh=None,
            return_word_box=None,
    ):
        return self.paddlex_pipeline.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
            return_word_box=return_word_box,
        )

    def predict(
            self,
            input,
            *,
            use_doc_orientation_classify=None,
            use_doc_unwarping=None,
            use_textline_orientation=None,
            text_det_limit_side_len=None,
            text_det_limit_type=None,
            text_det_thresh=None,
            text_det_box_thresh=None,
            text_det_unclip_ratio=None,
            text_rec_score_thresh=None,
            return_word_box=None,
    ):
        return list(
            self.predict_iter(
                input,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_limit_type=text_det_limit_type,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                text_rec_score_thresh=text_rec_score_thresh,
                return_word_box=return_word_box,
            )
        )

    @classmethod
    def get_cli_subcommand_executor(cls):
        return

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_name": self._params[
                "doc_orientation_classify_model_name"],
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_dir": self._params[
                "doc_orientation_classify_model_dir"],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name": self._params[
                "doc_unwarping_model_name"],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir": self._params["doc_unwarping_model_dir"],
            "SubModules.TextDetection.model_name": self._params["text_detection_model_name"],
            "SubModules.TextDetection.model_dir": self._params["text_detection_model_dir"],
            "SubModules.TextLineOrientation.model_name": self._params["textline_orientation_model_name"],
            "SubModules.TextLineOrientation.model_dir": self._params["textline_orientation_model_dir"],
            "SubModules.TextLineOrientation.batch_size": self._params["textline_orientation_batch_size"],
            "SubModules.TextRecognition.model_name": self._params["text_recognition_model_name"],
            "SubModules.TextRecognition.model_dir": self._params["text_recognition_model_dir"],
            "SubModules.TextRecognition.batch_size": self._params["text_recognition_batch_size"],
            "SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params["use_doc_orientation_classify"],
            "SubPipelines.DocPreprocessor.use_doc_unwarping": self._params["use_doc_unwarping"],
            "use_doc_preprocessor": self._params["use_doc_orientation_classify"] or self._params["use_doc_unwarping"],
            "use_textline_orientation": self._params["use_textline_orientation"],
            "SubModules.TextDetection.limit_side_len": self._params["text_det_limit_side_len"],
            "SubModules.TextDetection.limit_type": self._params["text_det_limit_type"],
            "SubModules.TextDetection.thresh": self._params["text_det_thresh"],
            "SubModules.TextDetection.box_thresh": self._params["text_det_box_thresh"],
            "SubModules.TextDetection.unclip_ratio": self._params["text_det_unclip_ratio"],
            "SubModules.TextDetection.input_shape": self._params["text_det_input_shape"],
            "SubModules.TextRecognition.score_thresh": self._params["text_rec_score_thresh"],
            "SubModules.TextRecognition.return_word_box": self._params["return_word_box"],
            "SubModules.TextRecognition.input_shape": self._params["text_rec_input_shape"]
        }
        return create_config_from_structure(STRUCTURE)

    def _get_ocr_model_names(self, lang, ppocr_version):
        LATIN_LANGS = [
            "af",
            "az",
            "bs",
            "cs",
            "cy",
            "da",
            "de",
            "es",
            "et",
            "fr",
            "ga",
            "hr",
            "hu",
            "id",
            "is",
            "it",
            "ku",
            "la",
            "lt",
            "lv",
            "mi",
            "ms",
            "mt",
            "nl",
            "no",
            "oc",
            "pi",
            "pl",
            "pt",
            "ro",
            "rs_latin",
            "sk",
            "sl",
            "sq",
            "sv",
            "sw",
            "tl",
            "tr",
            "uz",
            "vi",
            "french",
            "german",
            "fi",
            "eu",
            "gl",
            "lb",
            "rm",
            "ca",
            "qu",
        ]
        ARABIC_LANGS = ["ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"]
        ESLAV_LANGS = ["ru", "be", "uk"]
        CYRILLIC_LANGS = [
            "ru",
            "rs_cyrillic",
            "be",
            "bg",
            "uk",
            "mn",
            "abq",
            "ady",
            "kbd",
            "ava",
            "dar",
            "inh",
            "che",
            "lbe",
            "lez",
            "tab",
            "kk",
            "ky",
            "tg",
            "mk",
            "tt",
            "cv",
            "ba",
            "mhr",
            "mo",
            "udm",
            "kv",
            "os",
            "bua",
            "xal",
            "tyv",
            "sah",
            "kaa",
        ]
        DEVANAGARI_LANGS = [
            "hi",
            "mr",
            "ne",
            "bh",
            "mai",
            "ang",
            "bho",
            "mah",
            "sck",
            "new",
            "gom",
            "sa",
            "bgc",
        ]
        SPECIFIC_LANGS = [
            "ch",
            "en",
            "korean",
            "japan",
            "chinese_cht",
            "te",
            "ka",
            "ta",
        ]

        if lang is None:
            lang = "ch"

        if ppocr_version is None:
            if (lang in [
                "ch",
                "chinese_cht",
                "en",
                "japan",
                "korean",
                "th",
                "el",
                "te",
                "ta",
            ]
                    + LATIN_LANGS
                    + ESLAV_LANGS
                    + ARABIC_LANGS
                    + CYRILLIC_LANGS
                    + DEVANAGARI_LANGS
            ):
                ppocr_version = "PP-OCRv5"
            elif lang in (SPECIFIC_LANGS):
                ppocr_version = "PP-OCRv3"
            else:
                # Unknown language specified
                return None, None

        if ppocr_version == "PP-OCRv5":
            rec_lang, rec_model_name = None, None
            if lang in ("ch", "chinese_cht", "japan"):
                rec_model_name = "PP-OCRv5_server_rec"
            elif lang == "en":
                rec_model_name = "en_PP-OCRv5_mobile_rec"
            elif lang in LATIN_LANGS:
                rec_lang = "latin"
            elif lang in ESLAV_LANGS:
                rec_lang = "eslav"
            elif lang in ARABIC_LANGS:
                rec_lang = "arabic"
            elif lang in CYRILLIC_LANGS:
                rec_lang = "cyrillic"
            elif lang in DEVANAGARI_LANGS:
                rec_lang = "devanagari"
            elif lang == "korean":
                rec_lang = "korean"
            elif lang == "th":
                rec_lang = "th"
            elif lang == "el":
                rec_lang = "el"
            elif lang == "te":
                rec_lang = "te"
            elif lang == "ta":
                rec_lang = "ta"

            if rec_lang is not None:
                rec_model_name = f"{rec_lang}_PP-OCRv5_mobile_rec"
            return "PP-OCRv5_server_det", rec_model_name

        elif ppocr_version == "PP-OCRv4":
            if lang == "ch":
                return "PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"
            elif lang == "en":
                return "PP-OCRv4_mobile_det", "en_PP-OCRv4_mobile_rec"
            else:
                return None, None
        else:
            # PP-OCRv3
            rec_lang = None
            if lang in LATIN_LANGS:
                rec_lang = "latin"
            elif lang in ARABIC_LANGS:
                rec_lang = "arabic"
            elif lang in CYRILLIC_LANGS:
                rec_lang = "cyrillic"
            elif lang in DEVANAGARI_LANGS:
                rec_lang = "devanagari"
            else:
                if lang in SPECIFIC_LANGS:
                    rec_lang = lang

            rec_model_name = None
            if rec_lang == "ch":
                rec_model_name = "PP-OCRv3_mobile_rec"
            elif rec_lang is not None:
                rec_model_name = f"{rec_lang}_PP-OCRv3_mobile_rec"
            return "PP-OCRv3_mobile_det", rec_model_name


def test_ocr():
    from pathlib import Path

    ocr, test_det, test_rec = PaddleOCR(), TextDetection(), TextRecognition()
    img_files = Path(r"")
    for img_file in img_files.iterdir():
        print(img_file)
        result = ocr.predict(str(img_file))
        print(result)
        result2 = test_det.predict(str(img_file))
        print("-" * 200)
        print(result2)
        result3 = test_rec.predict(str(img_file))
        print("-" * 200)
        print(result3)
        print("=" * 200)
        # break


if __name__ == "__main__":
    test_ocr()
