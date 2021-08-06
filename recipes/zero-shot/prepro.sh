#!/bin/bash
export systemName=pmIndia.star.9

export WORKDIR=$HERE/data/lrt
export BASEDIR=$WORKDIR/$systemName/
export BPESIZE=40000
export PREPRO_TYPE=${BPESIZE}_sentencepiece

export INDIC_TOKENIZER="bash $OPTDIR/flores/scripts/indic_norm_tok.sh"
export SENTENCE_PIECE=true

PREPRO_DIR=prepro_${PREPRO_TYPE}

$SCRIPTDIR/scripts/defaultPreprocessor/Train.sh orig $PREPRO_DIR

for sl in en te kn ml bn gu hi mr or pa; do
        for tl in en te kn ml bn gu hi mr or pa; do
                if [ "$sl" != "$tl" ]; then
                        $SCRIPTDIR/scripts/defaultPreprocessor/Translate.sh $sl-$tl $PREPRO_DIR
                fi
        done
done
