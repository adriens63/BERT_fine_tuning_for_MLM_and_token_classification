# analysis_of_textual_job_descriptions_with_Pole_Emploi

# Etapes du projet 
Je note içi les étapes du projet, elles sont INDICATIVES et vous pouvez les modifier ou les changer d'ordre à votre guise

## Etape 1 : Statistiques descriptives 
- Longueur des textes, mots récurrents, spellchecker ... 

## Etape 2 : 
- Construction du modèle CamemBERT (librairie Hugging Face ?) 
- Entraînement sur la tâche de Masking dans un premier temps pour faire fonctionner le modèle sans avoir à labelliser 
- Autre option : Entraîner sur une tâche de prédiction (salaire, genre pour les CV)
- Faire un résumé du fonctionnement d'un BERT (servira pour le rapport final :) ) 
- Faire un résumé des bonne pratiques d'entraînement d'un BERT (Important !) 

## Etape 3 : 
- Mettre en place la stratégie de labellisation (quoi labelliser ?) 
- A première vue, une stratégie de labellisation binaire [soft-skill, autre] (réfléchir à d'autres options) 
- Réfléchir à la stratégie de construction du sample d'entraînement, favoriser la diversité des offres dans le sample (construire une mesure de distance entre les offres) 

## Etape 4 : 
- Mettre en place une application de labbelisation pour faciliter le partage (pas de nom en tête, voir ce qui existe)

## Etape 5 : 
- Labelliser les offres et les accroches selon la stratégie choisie (1000 pour démarrer)

## Etape 6 : 
- Entraînement du modèle
- Analyse des performances du modèle sur sample test : matrices de confusion, ou autres métriques appropriées à chercher dans les papiers existants et les bonnes pratiques 
- Tune des hyper paramètres 

## Etape 7 : 
- Monter une modèle de classification non-supervisé qui servira à classifier les soft-skill extraits des textes. On pourra construire un registre de soft-skill pertinents "data-driven"

## Etape 8 : 
- Construire sur votre Git une application de démonstration (input : texte, output : soft-skills demandés par les recruteurs / signalés pas les demandeurs) 

## Etape 9 : 
- Analyse et conclusions économiques (on laisera libre cours à notre imagination :p )
