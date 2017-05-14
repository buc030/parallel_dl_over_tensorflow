
python main.py -n 1 -history 0 -lr 0.7425 -sesop_batch_mult 2 -NORMALIZE_DIRECTIONS True -DISABLE_VECTOR_BREAKING True

for n in 2
do

    for h in 16
    do
        python main.py -n $n -history $h -lr 0.7425 -sesop_batch_mult 2 -NORMALIZE_DIRECTIONS True -DISABLE_VECTOR_BREAKING True
    done

    for h in 16
    do
        python main.py -n $n -history $h -lr 0.7425 -sesop_batch_mult 2 -NORMALIZE_DIRECTIONS True -DISABLE_VECTOR_BREAKING False
    done

    exit()

    for h in 16
    do
        python main.py -n $n -history $h -lr 0.7425 -sesop_batch_mult 2 -NORMALIZE_DIRECTIONS False -DISABLE_VECTOR_BREAKING True
    done

    for h in 16
    do
        python main.py -n $n -history $h -lr 0.7425 -sesop_batch_mult 2 -NORMALIZE_DIRECTIONS False -DISABLE_VECTOR_BREAKING False
    done
done

