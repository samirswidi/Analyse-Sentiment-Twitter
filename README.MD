Projet réalisé par Souidi Samir 
Notice :
1) Utiliser databricks pour l'affichage des graphiques 
2) importer la data set tweets1-1.csv dans /FileStore/tables/tweets1-1.csv
 (remarque dataset tweets1-1.csv se trouve dans le dossier dataset)
 3) Installer la bibliothèque twitter4j-core
  • pour installer cette biblothèque suivre le démarche suivant 
  ♦ choisir votre cluster actuel 
  ♦ choisir le menu libraries
  ♦ cliquer sur install new 
  ♦Cocher Maven
  ♦ cliquer sur Search packages ( maven coordinates)
  ♦ dans liste déroulante choisir maven central
  ♦ dans la zone de recherche taper org.twitter4j et   choisir twitter4j-core
4) Vous pouvez exécuter maintenat le code
5) Lien pour démo: https://www.youtube.com/watch?v=6YDpBHFYeTE
---------------------------------------------------------------------------

Pour utiliser spark-shell
1) modifier cette ligne 
   var tweets_csv = spark.read.format("csv").schema(schema).load("G:\\votre emplacement de fichier tweets1.csv\\tweets1.csv")
2) ouvrir invite de commande et taper spark-shell
3) :load G:/emplacement de votre code fichier scala/sent.scala
4) Lien pour démo: https://www.youtube.com/watch?v=U2DhCi98w5c
