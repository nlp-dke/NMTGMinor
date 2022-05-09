#!/bin/bash
source ./recipes/zero-shot/config.sh
set -eu


export MOSES=~/opt/mosesdecoder
export DATA_DIR= # path to tokenized test data
export BASEDIR=	# path to model & orig data
export name=$1 	# model name

LAN="it nl ro en"

mkdir $BASEDIR/data/$name -p

for sl in $LAN; do
for tl in $LAN; do

if [[ ! "$sl" == "$tl" ]]; then

pred_src=$DATA_DIR/tst2017$sl-$tl.real.s

echo $pred_src

out=$BASEDIR/data/$name/$sl-$tl.pred

echo "Input:" $pred_src 
echo "Output: " $out

bos='#'${tl^^}

python3 -u $NMTDIR/translate.py -gpu $GPU \
       -model $BASEDIR/model/$name/model.pt \
       -src $pred_src \
       -batch_size 128 -verbose \
       -beam_size 4 -alpha 1.0 \
       -normalize \
       -output $out \
       -fast_translate \
       -src_lang $sl \
       -tgt_lang $tl \
       -bos_token $bos

        # postprocess output
        sed -e "s/@@ //g" $out  | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' -e 's/ - /-/g' | sed -e "s/ '/'/g" | sed -e "s/ '/'/g" | sed -e "s/%- / -/g" | sed -e "s/ -%/- /g" | perl -nle 'print ucfirst' > $out.tok

        $MOSESDIR/scripts/tokenizer/detokenizer.perl -l $tl < $out.tok > $out.detok
        $MOSESDIR/scripts/recaser/detruecase.perl < $out.detok > $out.pt
	
	echo '===========================================' $sl $tl
	# Evaluate against original reference  
	cat $out.pt | sacrebleu $BASEDIR/data/orig/eval/tst2017$sl-$tl.real/tst2017$sl-$tl.real.$tl
	cat $out.pt | sacrebleu $BASEDIR/data/orig/eval/tst2017$sl-$tl.real/tst2017$sl-$tl.real.$tl > $BASEDIR/data/$name/$sl-$tl.test.res
fi

done
done
