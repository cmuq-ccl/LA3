rf=bm25; qs=wt13; s=52m; np=128; k=1000; bd=/datasets/carnegraph; rd=$bd/results/carnegraph/$rf/clueweb12/catb/$s/wt13; res=$rd/$np.$k; grep -w idx $res | awk '{print int((NR-1)/1000)+251, "Q0", $6, (NR-1)%1000+1, $8, "LA3"}' | sed 's/://g' > $res.pp; $bd/systems/LA3/efficiency.py $res.pp
