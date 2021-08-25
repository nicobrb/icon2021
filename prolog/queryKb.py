import pandas as pd
import pyswip as psw
from os import path
from sys import argv

from belief_network import BeliefNet

pl = psw.Prolog()


def print_help():
    print("insert a query to submit to KB")
    print("write 'assert' to write a new clause in KB")
    print("write 'quit' to exit")


def main():
    if not path.isfile(argv[1]):
        print("file not found or wrong directory, returning")
        return

    print("loading knowledge base...")
    pl.consult(argv[1])
    pd.set_option('display.max_rows', 3000, 'display.max_columns', 10)

    while True:
        print("insert query:")
        query = input()
        if query == 'ailog':
            pl.consult("./datasets/ailog2.pl")
        elif query == 'assert':
            print('write assertion without final point')
            try:
                pl.assertz(input())
            except Exception as ex:
                print("error: ", str(ex))
        elif query == 'satisfaction':
            satisfaction()
        elif query == 'quit':
            break
        elif query == 'help':
            print_help()
        else:
            try:
                # la risposta ad una query è una [] se false, [{}] se true, [{var:val},...,{var:val}] nel caso ci
                # siano più soluzioni

                """if query.__contains__("similar_to"):
                    rooms = query[query.find("similar_to("):][11:]
                    print(rooms)"""

                resultSet = pl.query(query)
                list_results = list(resultSet)

                if not list_results:
                    print("false.")
                elif len(list_results) == 1 and len(list_results[0]) == 0:
                    print("true.")
                else:
                    df = pd.DataFrame(list_results)

                    todrop = set()
                    for index, row in df.iterrows():
                        for col in df.columns:
                            if isinstance(row[col], psw.Variable):
                                todrop.add(index)
                                break
                    df.drop(index=todrop, inplace=True)
                    print(df)

            except Exception as e:
                print("Error" + str(e))


def satisfaction():
    favorite_features = []
    print("write your preferences, separated by comma:")
    favorite_features = input().split(',')

    print("write query (room id variable is Room):")
    query = 'room(Room),' + input()
    try:
        # la risposta ad una query è una [] se false, [{}] se true, [{var:val},...,{var:val}] nel caso ci
        # siano più soluzioni

        resultSet = pl.query(query)
        list_results = list(resultSet)

        if not list_results:
            print("false.")
        elif len(list_results) == 1 and len(list_results[0]) == 0:
            print("true.")
        else:
            if len(list_results) > 10:
                df = pd.DataFrame(list_results[:10])
            else:
                df = pd.DataFrame(list_results)

            ids = list(df['Room'])
            bn = BeliefNet(ids,favorite_features) #BeliefNetwork(ids, favorite_features)
            results = bn.compute_probabilities()
            df['satisfaction_prob'] = results
            df.sort_values('satisfaction_prob', inplace=True, ascending=False)
            # prob_column = compute_probability(
            # df = pd.concat([pd,prob_column],axis=1)

            print(df)

    except Exception as e:
        print("Error" + str(e))


main()
# main("../datasets/kb.pl")
