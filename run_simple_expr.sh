
python main.py -n 1 -history 0 -lr 0.06 -sesop_batch_mult 10 -NORMALIZE_DIRECTIONS True -DISABLE_VECTOR_BREAKING True

for n in 1 2 4
do


############# DONE ################
#    for h in 1 2 4 8 16
#    do
#        python main.py -n $n -history $h -lr 0.06 -sesop_batch_mult 10 -NORMALIZE_DIRECTIONS True -DISABLE_VECTOR_BREAKING True
#    done


    for h in 1 2 4 8
    do
        python main.py -n $n -history $h -lr 0.06 -sesop_batch_mult 10 -NORMALIZE_DIRECTIONS True -DISABLE_VECTOR_BREAKING False
    done

    exit()

    for h in 1 2 4 8
    do
        python main.py -n $n -history $h -lr 0.06 -sesop_batch_mult 10 -NORMALIZE_DIRECTIONS False -DISABLE_VECTOR_BREAKING True
    done

    for h in 1 2 4 8
    do
        python main.py -n $n -history $h -lr 0.06 -sesop_batch_mult 10 -NORMALIZE_DIRECTIONS False -DISABLE_VECTOR_BREAKING False
    done
done

