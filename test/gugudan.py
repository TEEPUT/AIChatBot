for i in range(2,9,4):
    for k in range(1,10):
        for j in range(4):
            print(i+j,'x',k,'=',(i+j)*k,end='\t')
        print()
    print()