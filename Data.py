import pickle
import numpy as np
d={};
def getData(fileName):
    count=0;
    c1=0
    c2=0;
    X=[];
    Y=[];
    print("Hello");
    with open(fileName) as f:
        line=f.readline();
        while(line):
            line=f.readline();
            data=line.split(";");
            length = len(data);
            if(length==17):
                row=[];
                for i in range(length):
                    if(i==0 or i==5 or i==9 or i==11 or i==12 or i==13 or i==14):
                       row.append(int(data[i]));
                    else:
                        k=data[i].split("'");
                        if(k[1] in d.keys()):
                            row.append(d[k[1]]);
                        else:
                            d[k[1]]=count;
                            count=count+1;
                            row.append(d[k[1]]);
                #0 5 9 11 12 13 14
                z=row[0:length-1];
                #X.append(row[0:length-1]);
                if(row[length-1]==d['no']):
                    z.append(0);
                    X.append(z);
                else:
                    z.append(1);

                    for j in range(6):
                        X.append(z);

    X1=np.array(X);
    np.random.shuffle(X1)

    X=[];
    a=len(X1[0])
    for i in range(len(X1)):
        X.append(X1[i][0:a-1])
        Y.append([X1[i][a-1]])

    return X,Y;


if __name__=="__main__":
    X,Y=getData('bank-full.csv');
    pickle.dump(X,open('X','wb'));
    print(len(X));
    pickle.dump(Y,open('Y','wb'));