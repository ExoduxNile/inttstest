# -*- coding: utf-8 -*-
import os
import traceback
import re
from typing import List, Union, overload
import warnings
from indextts.utils.common import tokenize_by_CJK_char, de_tokenized_by_CJK_char
from sentencepiece import SentencePieceProcessor


class TextNormalizer:
    def __init__(self):
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": " ",
            "·": "-",
            "、": ",",
            "...": "…",
            ",,,": "…",
            "，，，": "…",
            "……": "…",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }
        self.zh_char_rep_map = {
            "$": ".",
            **self.char_rep_map,
        }

    def match_email(self, email):
        pattern = r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$"
        return re.match(pattern, email) is not None

    PINYIN_TONE_PATTERN = r"([bmnpqdfghjklzcsxwy]?h?[aeiouüv]{1,2}[ng]*|ng)([1-5])"
    NAME_PATTERN = r"[\u4e00-\u9fff]+([-·—][\u4e00-\u9fff]+){1,2}"

    def use_chinese(self, s):
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", s))
        has_alpha = bool(re.search(r"[a-zA-Z]", s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True
        has_pinyin = bool(re.search(self.PINYIN_TONE_PATTERN, s, re.IGNORECASE))
        return has_pinyin

    def load(self):
        # Use self as fallback normalizer
        self.zh_normalizer = self
        self.en_normalizer = self

    def normalize(self, text: str) -> str:
        if not self.zh_normalizer or not self.en_normalizer:
            self.load()
            
        if self.use_chinese(text):
            replaced_text, pinyin_list = self.save_pinyin_tones(text.rstrip())
            replaced_text, original_name_list = self.save_names(replaced_text)
            
            # Use basic normalization
            result = self.basic_normalize(replaced_text)
            
            result = self.restore_names(result, original_name_list)
            result = self.restore_pinyin_tones(result, pinyin_list)
            pattern = re.compile("|".join(re.escape(p) for p in self.zh_char_rep_map.keys()))
            result = pattern.sub(lambda x: self.zh_char_rep_map[x.group()], result)
        else:
            result = self.basic_normalize(text)
            pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
            result = pattern.sub(lambda x: self.char_rep_map[x.group()], result)
        return result

    def basic_normalize(self, text: str) -> str:
        """Basic text normalization using regex patterns"""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def correct_pinyin(self, pinyin: str):
        if pinyin[0] not in "jqxJQX":
            return pinyin
        pattern = r"([jqx])[uü](n|e|an)*(\d)"
        repl = r"\g<1>v\g<2>\g<3>"
        pinyin = re.sub(pattern, repl, pinyin, flags=re.IGNORECASE)
        return pinyin.upper()

    def save_names(self, original_text):
        name_pattern = re.compile(self.NAME_PATTERN, re.IGNORECASE)
        original_name_list = re.findall(name_pattern, original_text)
        if len(original_name_list) == 0:
            return (original_text, None)
        original_name_list = list(set("".join(n) for n in original_name_list))
        transformed_text = original_text
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(name, f"<n_{number}>")
        return transformed_text, original_name_list

    def restore_names(self, normalized_text, original_name_list):
        if not original_name_list or len(original_name_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, name in enumerate(original_name_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(f"<n_{number}>", name)
        return transformed_text

    def save_pinyin_tones(self, original_text):
        origin_pinyin_pattern = re.compile(self.PINYIN_TONE_PATTERN, re.IGNORECASE)
        original_pinyin_list = re.findall(origin_pinyin_pattern, original_text)
        if len(original_pinyin_list) == 0:
            return (original_text, None)
        original_pinyin_list = list(set("".join(p) for p in original_pinyin_list))
        transformed_text = original_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            transformed_text = transformed_text.replace(pinyin, f"<pinyin_{number}>")
        return transformed_text, original_pinyin_list

    def restore_pinyin_tones(self, normalized_text, original_pinyin_list):
        if not original_pinyin_list or len(original_pinyin_list) == 0:
            return normalized_text
        transformed_text = normalized_text
        for i, pinyin in enumerate(original_pinyin_list):
            number = chr(ord("a") + i)
            pinyin = self.correct_pinyin(pinyin)
            transformed_text = transformed_text.replace(f"<pinyin_{number}>", pinyin)
        return transformed_text


class TextTokenizer:
    def __init__(self, vocab_file: str, normalizer: TextNormalizer = None):
        self.vocab_file = vocab_file
        self.normalizer = normalizer

        if self.vocab_file is None:
            raise ValueError("vocab_file is None")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"vocab_file {self.vocab_file} does not exist")
        if self.normalizer:
            self.normalizer.load()
        # 加载词表
        self.sp_model = SentencePieceProcessor(model_file=self.vocab_file)

        self.pre_tokenizers = [
            # 预处理器
            tokenize_by_CJK_char,
        ]

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def pad_token(self):
        return None

    @property
    def bos_token(self):
        return "<s>"

    @property
    def eos_token(self):
        return "</s>"

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    @property
    def eos_token_id(self):
        return 1

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def special_tokens_map(self):
        return {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab

    @overload
    def convert_ids_to_tokens(self, ids: int) -> str: ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]: ...

    def convert_ids_to_tokens(self, ids: Union[List[int], int]):
        return self.sp_model.IdToPiece(ids)

    def convert_tokens_to_ids(self, tokens: Union[List[str], str]) -> List[int]:
        if isinstance(tokens, str):
            tokens = [tokens]
        return [self.sp_model.PieceToId(token) for token in tokens]

    def tokenize(self, text: str) -> List[str]:
        return self.encode(text, out_type=str)

    def encode(self, text: str, **kwargs):
        if len(text) == 0:
            return []
        if len(text.strip()) == 1:
            return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)
        # 预处理
        if self.normalizer:
            text = self.normalizer.normalize(text)
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                text = pre_tokenizer(text)
        return self.sp_model.Encode(text, out_type=kwargs.pop("out_type", int), **kwargs)

    def batch_encode(self, texts: List[str], **kwargs):
        # 预处理
        if self.normalizer:
            texts = [self.normalizer.normalize(text) for text in texts]
        if len(self.pre_tokenizers) > 0:
            for pre_tokenizer in self.pre_tokenizers:
                texts = [pre_tokenizer(text) for text in texts]
        return self.sp_model.Encode(texts, out_type=kwargs.pop("out_type", int), **kwargs)

    def decode(self, ids: Union[List[int], int], do_lower_case=False, **kwargs):
        if isinstance(ids, int):
            ids = [ids]
        decoded = self.sp_model.Decode(ids, out_type=kwargs.pop("out_type", str), **kwargs)
        return de_tokenized_by_CJK_char(decoded, do_lower_case=do_lower_case)

    @staticmethod
    def split_sentences_by_token(
        tokenized_str: List[str], split_tokens: List[str], max_tokens_per_sentence: int
    ) -> List[List[str]]:
        """
        将tokenize后的结果按特定token进一步分割
        """
        sentences: List[List[str]] = []
        current_sentence = []
        for i in range(len(tokenized_str)):
            token = tokenized_str[i]
            current_sentence.append(token)
            if token in split_tokens:
                if len(current_sentence) == 1:
                    # 如果当前tokens只有一个，且是切分符号，则忽略这条句子
                    pass
                elif len(current_sentence) == 2 and current_sentence[0] == '▁':
                    # 如果当前tokens只有两个，且仅有切分符号，则忽略这条句子
                    pass
                elif len(current_sentence) <= max_tokens_per_sentence:
                    if i < len(tokenized_str) - 1:
                        if tokenized_str[i + 1] in ["'", "▁'"]:
                            # 后续token是'，则不切分
                            current_sentence.append(tokenized_str[i + 1])
                            i += 1

                    sentences.append(current_sentence)
                else:
                    # 如果当前tokens的长度超过最大限制
                    if "," in current_sentence or "▁," in current_sentence: 
                        # 如果当前tokens中有,，则按,分割
                        sub_sentences = TextTokenizer.split_sentences_by_token(
                            current_sentence, [",", "▁,"], max_tokens_per_sentence=max_tokens_per_sentence
                        )
                    elif "-" in current_sentence:
                        # 没有,，则按-分割
                        sub_sentences = TextTokenizer.split_sentences_by_token(
                            current_sentence, ["-"], max_tokens_per_sentence=max_tokens_per_sentence
                        )
                    else:
                        # 按照长度分割
                        sub_sentences = [
                            current_sentence[:max_tokens_per_sentence],
                            current_sentence[max_tokens_per_sentence:],
                        ]
                        warnings.warn(
                            f"The tokens length of sentence exceeds limit: {max_tokens_per_sentence}, "
                            f"Tokens in sentence: {current_sentence}."
                            "Maybe unexpected behavior",
                            RuntimeWarning,
                        )
                    sentences.extend(sub_sentences)
                current_sentence = []
        if len(current_sentence) > 0:
            sentences.append(current_sentence)
        # 如果相邻的句子加起来长度小于最大限制，则合并
        merged_sentences = []
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            if len(merged_sentences) == 0:
                merged_sentences.append(sentence)
            elif len(merged_sentences[-1]) + len(sentence) <= max_tokens_per_sentence:
                merged_sentences[-1] = merged_sentences[-1] + sentence
            else:
                merged_sentences.append(sentence)
        return merged_sentences

    punctuation_marks_tokens = [
        ".",
        "!",
        "?",
        "▁.",
        # "▁!", # unk
        "▁?",
        "▁...", # ellipsis
    ]
    def split_sentences(self, tokenized: List[str], max_tokens_per_sentence=120) -> List[List[str]]:
        return TextTokenizer.split_sentences_by_token(
            tokenized, self.punctuation_marks_tokens, max_tokens_per_sentence=max_tokens_per_sentence
        )


if __name__ == "__main__":
    # 测试程序

    text_normalizer = TextNormalizer()

    cases = [
        "IndexTTS 正式发布1.0版本了，效果666",
        "晕XUAN4是一种GAN3觉",
        "我爱你！",
        "I love you!",
        "“我爱你”的英语是“I love you”",
        "2.5平方电线",
        "共465篇，约315万字",
        "2002年的第一场雪，下在了2003年",
        "速度是10km/h",
        "现在是北京时间2025年01月11日 20:00",
        "他这条裤子是2012年买的，花了200块钱",
        "电话：135-4567-8900",
        "1键3连",
        "他这条视频点赞3000+，评论1000+，收藏500+",
        "这是1024元的手机，你要吗？",
        "受不liao3你了",
        "“衣裳”不读衣chang2，而是读衣shang5",
        "最zhong4要的是：不要chong2蹈覆辙",
        "不zuo1死就不会死",
        "See you at 8:00 AM",
        "8:00 AM 开会",
        "Couting down 3, 2, 1, go!",
        "数到3就开始：1、2、3",
        "This sales for 2.5% off, only $12.5.",
        "苹果于2030/1/2发布新 iPhone 2X 系列手机，最低售价仅 ¥12999",
        "这酒...里...有毒...",
        # 异常case
        "只有,,,才是最好的",
        # 人名
        "约瑟夫·高登-莱维特（Joseph Gordon-Levitt is an American actor）",
        "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），美国商业经理、工业工程师和工业开发商，现任苹果公司首席执行官。",
        # 长句子
        "《盗梦空间》是由美国华纳兄弟影片公司出品的电影，由克里斯托弗·诺兰执导并编剧，莱昂纳多·迪卡普里奥、玛丽昂·歌迪亚、约瑟夫·高登-莱维特、艾利奥特·佩吉、汤姆·哈迪等联袂主演，2010年7月16日在美国上映，2010年9月1日在中国内地上映，2020年8月28日在中国内地重映。影片剧情游走于梦境与现实之间，被定义为“发生在意识结构内的当代动作科幻片”，讲述了由莱昂纳多·迪卡普里奥扮演的造梦师，带领特工团队进入他人梦境，从他人的潜意识中盗取机密，并重塑他人梦境的故事。",
    ]
    # 测试分词器
    tokenizer = TextTokenizer(
        vocab_file="checkpoints/bpe.model",
        normalizer=text_normalizer,
    )

    codes = tokenizer.batch_encode(
        cases,
        out_type=int,
    )

    print(f"vocab_size: {tokenizer.vocab_size}")
    # print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    print(f"eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"unk_token: {tokenizer.unk_token}, unk_token_id: {tokenizer.unk_token_id}")
    # 不应该有 unk_token_id
    for t in set([*TextTokenizer.punctuation_marks_tokens, ",", "▁,", "-", "▁..."]):
        tokens = tokenizer.convert_tokens_to_ids(t)
        if tokenizer.unk_token_id in tokens:
            print(f"Warning: {t} is unknown token")
        print(f"`{t}`", "->", tokens, "->", tokenizer.convert_ids_to_tokens(tokens))
    for ch in set(tokenizer.normalizer.zh_char_rep_map.values()):
        # 测试 normalize后的字符能被分词器识别
        print(f"`{ch}`", "->", tokenizer.sp_model.Encode(ch, out_type=str))
        print(f"` {ch}`", "->", tokenizer.sp_model.Encode(f" {ch}", out_type=str))
    for i in range(len(cases)):
        print(f"原始文本: {cases[i]}")
        print(f"Normalized: {text_normalizer.normalize(cases[i])}")
        tokens = tokenizer.tokenize(cases[i])
        print(f"Tokenzied: {tokens}")
        sentences = tokenizer.split_sentences(tokens, max_tokens_per_sentence=100)
        print("Splitted sentences count:", len(sentences))
        if len(sentences) > 1:
            for j in range(len(sentences)):
                print(f"  {j}, count:", len(sentences[j]), ", tokens:", "".join(sentences[j]))
        #print(f"Token IDs (first 10): {codes[i][:10]}")
        if tokenizer.unk_token in codes[i]:
            print(f"Warning: `{cases[i]}` contains UNKNOWN token")
        print(f"Decoded: {tokenizer.decode(codes[i], do_lower_case=True)}")
        print("-" * 50)
