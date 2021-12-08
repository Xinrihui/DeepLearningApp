
1.下载 moses

git clone https://github.com/moses-smt/mosesdecoder

2.运行 bleu 评价脚本

在 linux 命令行下执行:

perl tools/mosesdecoder/scripts/generic/multi-bleu.perl dataset/WMT-14-English-Germa/newstest2014.de < outs/candidates.txt

