import pandas as pd
from os import path, remove
from sys import argv


def createKB(dataframe, kbpath, max_rows=-1):
    kb = open(kbpath, "a")
    pd.set_option('display.max_rows', 1000)

    print('creating facts from dataset in KB')

    feature_ignored = ['description']
    string_features = ['name', 'description', 'neighbourhood_cleansed',
                       'neighbourhood_group_cleansed', 'property_type',
                       'room_type', 'n_cluster']
    boolean_features = ['has_availability', 'instant_bookable', 'host_has_profile_pic',
                        'host_identity_verified', 'host_is_superhost']
    list_features = ['host_verifications', 'amenities']

    properties = dataframe.columns
    for i in range(0, len(properties)):
        cont = 0
        try:
            for row in dataframe.iterrows():
                cont = cont + 1
                if 0 <= max_rows < cont:
                    break

                data = row[1]

                if properties[i] in feature_ignored:
                    continue
                elif properties[i] == 'id':
                    # asserisco l'esistenza di una stanza con il suo id
                    kb.write('room(' + str(data["id"]) + ').\n')
                elif properties[i] in list_features:
                    cleaned_list = data[properties[i]].replace('[', '') \
                        .replace(']', "").replace('"', '').replace("'", '').replace("\\", '').split(",")
                    for x in cleaned_list:
                        kb.write(properties[i] + "(" + str(data["id"]) + ',"' + str(x).lower().strip() + '").\n')
                elif properties[i] in string_features:
                    to_write = str(data[properties[i]]).lower() \
                        .replace("<b>", "").replace("</b>", "").replace("<br />", "").replace("\\", '').strip()
                    kb.write(properties[i] + "(" + str(data["id"]) + ',"' + to_write + '").\n')
                elif properties[i] in boolean_features:
                    if str(data[properties[i]]) == 't':
                        kb.write(properties[i] + '(' + str(data["id"]) + ').\n')
                else:
                    if data[properties[i]] != "nan":
                        kb.write(
                            properties[i] + "(" + str(data["id"]) + ',' + str(data[properties[i]]).strip() + ').\n')
        except Exception as e:
            print("Exception with row " + str(e))
    kb.close()


def createRules(kb_path):
    #creazione delle regole all'interno della base di conoscenza

    kb = open(kb_path, "a")
    print('creating rules in KB')

    kb.write('room_for_couples(X) :- bedrooms(X,1.0), room_type(X,"private room").\n')
    kb.write('is_available(X) :- has_availability(X), instant_bookable(X).\n')
    kb.write('room_for_family(Room,Children) :- beds(Room,Beds), Beds is (1.0+Children).\n')
    kb.write('connections(X,Y) :- amenities(X,Y),member(Y,["wifi","cable tv"]).\n')
    kb.write('class("luxury",P) :- P>700.0.\n')
    kb.write('class("expensive",P) :- P>200.0,P=<700.0 .\n')
    kb.write('class("medium",P) :- P>50.0,P=<200.0 .\n')
    kb.write('class("low cost",P) :- P>30.0,P=<50.0 .\n')
    kb.write('class("cheap",P) :- P=<30.0 .\n')
    kb.write('class_no("expensive",4).\n')
    kb.write('class_no("luxury",5).\n')
    kb.write('class_no("medium",3).\n')
    kb.write('class_no("low cost", 2).\n')
    kb.write('class_no("cheap",1).\n')
    kb.write('higher_class(C1,C2) :- class_no(C1,N1), class_no(C2,N2), N1>N2.\n')
    kb.write('lower_class(C1,C2) :- higher_class(C2,C1).\n')
    kb.write('similar_room(X,Y) :- n_cluster(X,C), n_cluster(Y,C).\n')
    kb.write('same_price_range(R1,R2) :- price(R1,P1), price(R2,P2), class(C1,P1), class(C2,P2), C1=C2.\n')
    kb.write('same_amenities(R1,R2,A) :- amenities(R1,A), amenities(R2,A).\n')
    kb.write('price_range(Room,Range) :- price(Room,Price), class(Range,Price).\n')

    kb.close()


def main():
    if not path.isfile(argv[1]):
        print("File not found or not valid")
        return

    dataframe = pd.read_csv(argv[1])

    if path.exists('../datasets/kb.pl'):
        remove('../datasets/kb.pl')
    createKB(dataframe, '../datasets/kb.pl')
    createRules('../datasets/kb.pl')


main()
# main("../datasets/prolog_with_clusters.csv")
