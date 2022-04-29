#!/bin/bash
source ../../config.sh

PREPRO_DIR=prepro_40000_sentencepiece

BASEDIR=/home/dliu/data/lrt/pmIndia.star.9
name=$1

for sl in te kn ml bn gu hi mr or pa en ; do
for tl in te kn ml bn gu hi mr or pa en ; do

mkdir $BASEDIR/data/$name -p

if [[ ! "$sl" == "$tl" ]]; then
pred_src=$BASEDIR/data/$PREPRO_DIR/eval/$sl-$tl.s
out=$BASEDIR/data/$name/$sl-$tl.pred

echo $pred_src $out
bos='#'${tl}'#'

python3 -u $NMTDIR/translate.py -gpu $GPU \
       -model $BASEDIR/model/$name/model.pt \
       -src $pred_src \
       -batch_size 256 -verbose \
       -beam_size 4 -alpha 1.0 \
       -normalize \
       -output $out \
       -fast_translate \
       -src_lang $sl \
       -tgt_lang $tl \
       -bos_token $bos

        sed -e "s/ //g" $out | sed -e "s/▁/ /g" | perl -nle 'print ucfirst' > $out.tok

        # postprocess output
        $MOSESDIR/scripts/tokenizer/detokenizer.perl -l $tl < $out.tok > $out.detok
        $MOSESDIR/scripts/recaser/detruecase.perl < $out.detok > $out.pt

        # scoring
        cat $out.pt | sacrebleu $BASEDIR/data/orig/eval/$sl-$tl/$sl-$tl.$tl -tok spm
fi
done
done
