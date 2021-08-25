# icon2021

Repository riguardante il progetto correlato all'esame di Ingegneria
della Conoscenza, anno accademico 2020/2021 (Bari)

<h1>Prima di iniziare</h1>

- Installare SWIProlog 

    `` https://www.swi-prolog.org/download/stable/bin/swipl-8.2.4-1.x64.exe.envelope``


- Clonare il progetto

    `` git clone https://github.com/robertogasbarro/icon2021.git``

- Creare l'ambiente virtuale 

    ``cd icon2021``

    ``python -m venv icon2021``


- Installare le dipendenze

    ``pip install -r requirements.txt``

<h1>Come eseguire gli script</h1>

<em>Si fa notare che l'ordine di esecuzione indicato è 
<strong>imperativo</strong> quantomeno per la prima run del 
codice: infatti i file generati dal preprocessing sono necessari
per l'esecuzione dei restanti script.</em>

<h3>Dataset Preprocessing</h3>
    
    python preprocessing/preprocessing.py ./datasets/trainingset.csv

<h3>K-Medoids Clustering</h3>
    
    python clustering/clustering.py ./datasets/preprocessed.csv [num_iterate] [num_clusters]

<h3>Creazione della KB</h3>

    python prolog/kbCreation.py ./datasets/prolog_with_clusters.csv

<h3>Query alla KB</h3>

    python prolog/queryKb.py ./datasets/kb.pl
