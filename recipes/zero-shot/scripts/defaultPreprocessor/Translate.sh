#!/bin/bash


set=$1
name=$2
origname=$3

if [ -z "$BASEDIR" ]; then
    BASEDIR=/
fi

if [ -z "$MOSESDIR" ]; then
    MOSESDIR=/opt/mosesdecoder/
fi

if [ -z "$BPEDIR" ]; then
    BPEDIR=/opt/subword-nmt/
fi

if [ -z "$SRC_INDIC" ]; then
    SRC_INDIC=false
fi

if [ -z "$origname" ]; then
    origname=orig
fi


mkdir -p $BASEDIR/data/${name}/eval
mkdir -p $BASEDIR/data/${name}/valid

##TOKENIZE
##SMARTCASE
##BPE
echo $BASEDIR/data/$origname/eval/$set/$set.$sl

xml=0
if [ -f $BASEDIR/data/$origname/eval/$set/IWSLT.$set/IWSLT.TED.$set.$sl-$tl.$sl.xml ]; then
    inFile=$BASEDIR/data/$origname/eval/$set/IWSLT.$set/IWSLT.TED.$set.$sl-$tl.$sl.xml
    xml=1
elif [ -f $BASEDIR/data/$origname/eval/$set/$set.$sl ]; then
    inFile=$BASEDIR/data/$origname/eval/$set/$set.$sl
    xml=0
fi

echo 'inFile' $inFile
echo $BASEDIR/model/${name}/truecase-model.s
echo $BASEDIR/model/${name}/voc.s

xmlcommand=""
if [ $xml -eq 1 ]; then

 cat $inFile | grep "<seg id" | sed -e "s/<[^>]*>//g" | \
    perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} | \
    $MOSESDIR/scripts/recaser/truecase.perl --model $BASEDIR/model/${name}/truecase-model.s | \
    $BPEDIR/apply_bpe.py -c $BASEDIR/model/${name}/codec --vocabulary $BASEDIR/model/${name}/voc.s --vocabulary-threshold 50 \
				  > $BASEDIR/data/${name}/eval/manualTranscript.$set.s
else

sl=${set:0:2}
if [ ! "$SENTENCE_PIECE" == true ]; then
cat $inFile | \
    perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} | \
    $MOSESDIR/scripts/recaser/truecase.perl --model $BASEDIR/model/${name}/truecase-model.s | \
    $BPEDIR/apply_bpe.py -c $BASEDIR/model/${name}/codec --vocabulary $BASEDIR/model/${name}/voc.s --vocabulary-threshold 50 \
				  > $BASEDIR/data/${name}/eval/$set.s
else

if [ "$sl" != 'ml' ] && [ "$sl" != 'te' ]  && [ "$sl" != 'kn' ] && [ "$sl" != 'ta' ] && [ "$sl" != 'bn' ] && [ "$sl" != 'gu' ] && [ "$sl" != 'hi' ] && [ "$sl" != 'mr' ] && [ "$sl" != 'or' ] && [ "$sl" != 'pa' ]; then
	cat $inFile | perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} | \
		$MOSESDIR/scripts/recaser/truecase.perl --model $BASEDIR/model/${name}/truecase-model.s > $BASEDIR/data/${name}/eval/$set.tok.s
else
	$INDIC_TOKENIZER ${sl} $inFile > $BASEDIR/data/${name}/eval/$set.tok.s
fi

spm_encode \
	--model=$BASEDIR/data/${name}/sentencepiece.bpe.model \
        --output_format=piece \
        --vocabulary_threshold 50 < $BASEDIR/data/${name}/eval/$set.tok.s > $BASEDIR/data/${name}/eval/$set.s

#python $FLORES_SCRIPTS/spm_encode.py \
#                --model $BASEDIR/data/${name}/sentencepiece.bpe.model \
#                --output_format=piece \
#                --inputs $BASEDIR/data/${name}/eval/$set.tok.s \
#                --outputs $BASEDIR/data/${name}/eval/$set.s

fi
fi
