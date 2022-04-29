#!/bin/bash

if [ -z "$BASEDIR" ]; then
    BASEDIR=/
fi

if [ -z "$MOSESDIR" ]; then
    MOSESDIR=/opt/mosesdecoder/
fi

if [ -z "$BPEDIR" ]; then
    BPEDIR=/opt/subword-nmt/
fi

if [ -z "$BPESIZE" ]; then
    BPESIZE=40000
fi

if [ -z "$SRC_INDIC" ]; then
    SRC_INDIC=false
fi

if [ -z "$PARA" ]; then
    PARA=parallel	
fi


input=$1
name=$2
echo $PARA

mkdir -p $BASEDIR/tmp/${name}/tok/train
mkdir -p $BASEDIR/tmp/${name}/tok/valid
mkdir -p $BASEDIR/tmp/${name}/sc/train
mkdir -p $BASEDIR/tmp/${name}/sc/valid
mkdir -p $BASEDIR/model/${name}
mkdir -p $BASEDIR/data/${name}/train
mkdir -p $BASEDIR/data/${name}/valid

echo "*** Tokenization" 
#Source
echo "" > $BASEDIR/tmp/${name}/corpus.tok.s

for f in $BASEDIR/data/${input}/$PARA/*\.s
do
### train
lan_pair="$(basename "$f")"
sl=${lan_pair:0:2}
if [ "$sl" != 'ml' ] && [ "$sl" != 'te' ]  && [ "$sl" != 'kn' ] && [ "$sl" != 'ta' ] && [ "$sl" != 'bn' ] && [ "$sl" != 'gu' ] && [ "$sl" != 'hi' ] && [ "$sl" != 'mr' ] && [ "$sl" != 'or' ] && [ "$sl" != 'pa' ]; then
	cat $f | perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} > $BASEDIR/tmp/${name}/tok/train/${f##*/}
#	cat $BASEDIR/tmp/${name}/tok/train/${f##*/} >> $BASEDIR/tmp/${name}/corpus.tok.s
else
	echo '*** Using indic tokenizer for src' $f
	$INDIC_TOKENIZER ${sl} $f > $BASEDIR/tmp/${name}/tok/train/${f##*/}
#	cat $BASEDIR/tmp/${name}/tok/train/${f##*/} >> $BASEDIR/tmp/${name}/corpus.tok.s
fi
cat $BASEDIR/tmp/${name}/tok/train/${f##*/} >> $BASEDIR/tmp/${name}/corpus.tok.s
done
### valid
for f in $BASEDIR/data/${input}/valid/*\.s
do
lan_pair="$(basename "$f")"
sl=${lan_pair:0:2}
if [ "$sl" != 'ml' ] && [ "$sl" != 'te' ]  && [ "$sl" != 'kn' ] && [ "$sl" != 'ta' ] && [ "$sl" != 'bn' ] && [ "$sl" != 'gu' ] && [ "$sl" != 'hi' ] && [ "$sl" != 'mr' ] && [ "$sl" != 'or' ] && [ "$sl" != 'pa' ]; then
	cat $f | perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} > $BASEDIR/tmp/${name}/tok/valid/${f##*/}
else
	echo '*** Using indic tokenizer for src' $f
	$INDIC_TOKENIZER ${sl} $f > $BASEDIR/tmp/${name}/tok/valid/${f##*/}
fi
done

#Target
echo "" > $BASEDIR/tmp/${name}/corpus.tok.t
### train
for f in $BASEDIR/data/${input}/$PARA/*\.t
do
lan_pair="$(basename "$f")"
tl=${lan_pair:3:2}
if [ "$tl" != 'ml' ] && [ "$tl" != 'te' ]  && [ "$tl" != 'kn' ] && [ "$tl" != 'ta' ] && [ "$tl" != 'bn' ] && [ "$tl" != 'gu' ] && [ "$tl" != 'hi' ] && [ "$tl" != 'mr' ] && [ "$tl" != 'or' ] && [ "$tl" != 'pa' ]; then
	cat $f | perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${tl} > $BASEDIR/tmp/${name}/tok/train/${f##*/}
#	cat $BASEDIR/tmp/${name}/tok/train/${f##*/} >> $BASEDIR/tmp/${name}/corpus.tok.t
else
	echo '*** Using indic tokenizer for tgt' $f
	$INDIC_TOKENIZER ${tl} $f > $BASEDIR/tmp/${name}/tok/train/${f##*/}
fi
cat $BASEDIR/tmp/${name}/tok/train/${f##*/} >> $BASEDIR/tmp/${name}/corpus.tok.t
done
### valid
for f in $BASEDIR/data/${input}/valid/*\.t
do
lan_pair="$(basename "$f")"
tl=${lan_pair:3:2}
if [ "$tl" != 'ml' ] && [ "$tl" != 'te' ]  && [ "$tl" != 'kn' ] && [ "$tl" != 'ta' ] && [ "$tl" != 'bn' ] && [ "$tl" != 'gu' ] && [ "$tl" != 'hi' ] && [ "$tl" != 'mr' ] && [ "$tl" != 'or' ] && [ "$tl" != 'pa' ]; then
	cat $f | perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${tl} > $BASEDIR/tmp/${name}/tok/valid/${f##*/}
else
	echo '*** Using indic tokenizer for tgt' $f
        $INDIC_TOKENIZER ${tl} $f > $BASEDIR/tmp/${name}/tok/valid/${f##*/}
fi
done

##SMARTCASE
echo "*** Learning Truecaser" 
if [ "$SRC_INDIC" != true ]; then
$MOSESDIR/scripts/recaser/train-truecaser.perl --model $BASEDIR/model/${name}/truecase-model.s --corpus $BASEDIR/tmp/${name}/corpus.tok.s
fi

$MOSESDIR/scripts/recaser/train-truecaser.perl --model $BASEDIR/model/${name}/truecase-model.t --corpus $BASEDIR/tmp/${name}/corpus.tok.t

if [ "$SRC_INDIC" != true ]; then
for set in valid train
do
for f in $BASEDIR/tmp/${name}/tok/$set/*\.s
do
cat $f | $MOSESDIR/scripts/recaser/truecase.perl --model $BASEDIR/model/${name}/truecase-model.s > $BASEDIR/tmp/${name}/sc/$set/${f##*/}
done
done

else # skip smartcasing
for set in valid train
do
for f in $BASEDIR/tmp/${name}/tok/$set/*\.s
do
cat $f > $BASEDIR/tmp/${name}/sc/$set/${f##*/}
done
done

fi

for set in valid train
do
for f in $BASEDIR/tmp/${name}/tok/$set/*\.t
do
cat $f | $MOSESDIR/scripts/recaser/truecase.perl --model $BASEDIR/model/${name}/truecase-model.t > $BASEDIR/tmp/${name}/sc/$set/${f##*/}
done
done

echo "" > $BASEDIR/tmp/${name}/corpus.sc.s
for f in $BASEDIR/tmp/${name}/sc/train/*\.s
do
cat $f >> $BASEDIR/tmp/${name}/corpus.sc.s
done

echo "" > $BASEDIR/tmp/${name}/corpus.sc.t
for f in $BASEDIR/tmp/${name}/sc/train/*\.t
do
cat $f >> $BASEDIR/tmp/${name}/corpus.sc.t
done

#BPE
echo "*** Learning BPE of size" $BPESIZE
if [ ! "$SENTENCE_PIECE" == true ]; then
	echo "*** BPE by subword-nmt"
	$BPEDIR/subword_nmt/learn_joint_bpe_and_vocab.py --input $BASEDIR/tmp/${name}/corpus.sc.s $BASEDIR/tmp/${name}/corpus.sc.t -s $BPESIZE -o $BASEDIR/model/${name}/codec --write-vocabulary $BASEDIR/model/${name}/voc.s $BASEDIR/model/${name}/voc.t

	for set in valid train
	do
		for f in $BASEDIR/tmp/${name}/sc/$set/*\.s
		do
		echo $f
		$BPEDIR/subword_nmt/apply_bpe.py -c $BASEDIR/model/${name}/codec --vocabulary $BASEDIR/model/${name}/voc.s --vocabulary-threshold 50 < $f > $BASEDIR/data/${name}/$set/${f##*/}
		done
	done

	for set in valid train
	do
		for f in $BASEDIR/tmp/${name}/sc/$set/*\.t
		do
		echo $f
		$BPEDIR/subword_nmt/apply_bpe.py -c $BASEDIR/model/${name}/codec --vocabulary $BASEDIR/model/${name}/voc.t --vocabulary-threshold 50 < $f > $BASEDIR/data/${name}/$set/${f##*/}
		done
	done

else
	echo "*** BPE by sentencepiece"
	spm_train \
	--input=$BASEDIR/tmp/${name}/corpus.sc.s,$BASEDIR/tmp/${name}/corpus.sc.t \
        --model_prefix=$BASEDIR/data/${name}/sentencepiece.bpe \
        --vocab_size=$BPESIZE \
        --character_coverage=1.0 \
        --model_type=bpe
	for set in valid train
	do
		for f in $BASEDIR/tmp/${name}/sc/$set/* #\.s
		do
		echo $f
		spm_encode \
        	--model=$BASEDIR/data/${name}/sentencepiece.bpe.model \
        	--output_format=piece \
		--vocabulary_threshold 50 < $f > $BASEDIR/data/${name}/$set/${f##*/}
		done
	done
fi

rm -r $BASEDIR/tmp/${name}/
