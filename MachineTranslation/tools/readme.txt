
1.下载 moses

git clone https://github.com/moses-smt/mosesdecoder


2.运行分词脚本


3.运行 bleu 评价脚本

在 linux 命令行下执行:

perl tools/mosesdecoder/scripts/generic/multi-bleu.perl dataset/WMT-14-English-Germa/newstest2014.de < outs/candidates.txt

注意确认 newstest2014.de 和 candidates.txt 中句子的数目一致
