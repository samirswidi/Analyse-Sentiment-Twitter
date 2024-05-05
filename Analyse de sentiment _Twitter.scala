// Databricks notebook source
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, Tokenizer, StopWordsRemover}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.types._

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.io.StdIn

// Create SparkSession
val spark = SparkSession.builder().appName("TweetSentimentAnalysis").getOrCreate()

// Define schema for tweets data
val schema = StructType(Seq(
  StructField("ItemID", LongType, true),
  StructField("Sentiment", IntegerType, true),
  StructField("SentimentSource", StringType, true),
  StructField("SentimentText", StringType, true)
))

// COMMAND ----------



  
var tweets_csv = spark.read.format("csv").schema(schema).load("/FileStore/tables/tweets1-1.csv")

tweets_csv.createOrReplaceTempView("tweets_temp_table")
display(spark.sql("select * from tweets_temp_table"))
tweets_csv.take(10).foreach(println)


// COMMAND ----------

val data = tweets_csv.select($"SentimentText", col("Sentiment").cast("Int").as("label"))

// Afficher les cinq premières lignes du DataFrame
val Array(trainingData, testingData) = data.randomSplit(Array(0.7, 0.3))

// Compte les lignes dans chaque division

val trainRows = trainingData.count()
val testRows = testingData.count()


// Calculer les proportions du graphe Pie
val totalRows = trainRows + testRows
val trainPercentage = trainRows.toDouble / totalRows * 100
val testPercentage = testRows.toDouble / totalRows * 100

// Afficher les proportions
println(s"Training Data: $trainPercentage%")
println(s"Testing Data: $testPercentage%")

// Import necessary libraries for plotting
import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML

// Define the HTML code for the pie chart
val html = s"""
  <div id="piechart"></div>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script>
    var data = [{
      values: [$trainPercentage, $testPercentage],
      labels: ['Training Data', 'Testing Data'],
      type: 'pie'
    }];
    var layout = {
      title: '-----Data Split---Training and testing Data',
    };
    Plotly.newPlot('piechart', data, layout);
  </script>
"""

// Display the pie chart
displayHTML(html)
// Print the results
println("Training data rows: " + trainRows + "; Testing data rows: " + testRows)




// creation Tokenizer :  (un outil de tokenisation) qui transforme le texte en une séquence de tokens (mots).
val tokenizer = new Tokenizer().setInputCol("SentimentText").setOutputCol("SentimentWords")


// Appliquer le tokenizer aux données d'entraînement
  // Tokenize training data
  val tokenizedTrain = tokenizer.transform(trainingData)    //Applique le tokenizer aux données d'entraînement (trainingData) pour obtenir les tokens (SentimentWords).

  // Limit and show results (without truncation, check Spark version for specific method)
  val limitedTrain = tokenizedTrain.limit(5)
  limitedTrain.show(false) // Check your Spark version for the appropriate argument
  val swr = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("MeaningfulWords")
  //Cette ligne crée un outil de suppression des mots vides (StopWordsRemover) qui supprime les mots inutiles (comme "et", "ou", "le") des tokens.

   val SwRemovedTrain = swr.transform(tokenizedTrain)
   //Applique le StopWordsRemover aux données d'entraînement tokenisées (tokenizedTrain) pour obtenir les tokens sans les mots vides.
   SwRemovedTrain.limit(5).show(false)


   /**/
   val hashTF = new HashingTF()
  .setInputCol(swr.getOutputCol)
  .setOutputCol("features")
//Crée un outil de transformation HashingTF qui convertit les tokens (MeaningfulWords) en caractéristiques numériques (vecteurs de caractéristiques).
// Appliquer le HashingTF aux données d'entraînement
val numericTrainData = hashTF.transform(SwRemovedTrain).select(
    $"label", $"MeaningfulWords", $"features")
//Applique HashingTF aux données d'entraînement pour obtenir les caractéristiques numériques (features).

// Afficher les résultats
// Afficher les premières lignes du DataFrame numericTrainData
numericTrainData.limit(5).show(false)
import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML

// Import other necessary packages
import org.apache.spark.sql.functions._

// Prepare the data
// Calculate the frequency of each label
val labelCounts = numericTrainData.groupBy($"label").count()

// Convert the DataFrame to a Scala Map
val dataMap = labelCounts.collect().map(row => (row.getInt(0), row.getLong(1))).toMap

// Define the HTML code for the pie chart
val html = s"""
  <div id="piechart"></div>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script>
    // Define the data for the pie chart
    var data = [{
      values: [${dataMap.values.mkString(", ")}],
      labels: [${dataMap.keys.mkString(", ")}],
      type: 'pie'
    }];

    // Define the layout (optional)
    var layout = {
      title: 'Numeric data training'
    };

    // Create the pie chart
    Plotly.newPlot('piechart', data, layout);
  </script>
"""

// Display the HTML using displayHTML
displayHTML(html)



// Créer un modèle de régression logistique
val lr = new LogisticRegression()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxIter(10)
  .setRegParam(0.01)

// Entraîner le modèle
val model = lr.fit(numericTrainData)

//Type d'algorithme : La régression logistique est un algorithme de classification. Bien que son nom contienne "régression", 
// elle est utilisée pour la classification binaire ou multiclasse.
// Fonctionnalité : L'algorithme de régression logistique modélise la relation entre les caractéristiques d'entrée (colonne features) 
// et une variable cible binaire ou catégorique (label) en utilisant une fonction logistique ou sigmoïde.
// La sortie du modèle est la probabilité d'appartenir à une classe particulière.

println("Entraînement terminé !")

// Faire des prédictions sur les données d'entraînement
val prediction = model.transform(numericTrainData)

// Sélectionner les colonnes requises
val predictionFinal = prediction.select(
    "MeaningfulWords", "prediction", "label")

// Afficher les premières lignes du DataFrame
predictionFinal.show(4, false)

// Compter les prédictions correctes
val correctPrediction = predictionFinal.filter(
    predictionFinal("prediction") === predictionFinal("label")).count()

// Compter le nombre total de données
val totalData = predictionFinal.count()

// Calculer l'exactitude
val accuracy = correctPrediction.toDouble / totalData

// Afficher les résultats
println(s"Prédictions correctes : $correctPrediction, données totales : $totalData, exactitude : $accuracy")
//graphe

// Calculate the proportions for the pie chart

val correct=correctPrediction.toDouble/totalData.toDouble;
println(correct);
val correctpredic = correctPrediction.toDouble / totalData.toDouble * 100
val incorrectpredic = (totalData.toDouble-correctPrediction.toDouble) / totalData.toDouble * 100

// Display the proportions
println(s"Training Data: $correctpredic%")
println(s"Testing Data: $incorrectpredic%")

// Import necessary libraries for plotting
import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML

// Define the HTML code for the pie chart
val html = s"""
  <div id="piechart"></div>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script>
    var data = [{
      values: [$correctpredic, $incorrectpredic],
      labels: ['Correct prediction', 'Incorrect prediction'],
      type: 'pie'
    }];
    var layout = {
      title: '-----Data Split---Training and testing Data',
    };
    Plotly.newPlot('piechart', data, layout);
  </script>
"""

// Display the pie chart
displayHTML(html)






//graphe 
import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML
import org.apache.spark.sql.functions._

// Calculate the frequency of each word
val wordCounts = SwRemovedTrain
  .select("MeaningfulWords")
  .rdd
  .flatMap(row => row.getAs[Seq[String]]("MeaningfulWords"))
  .map(word => (word, 1))
  .reduceByKey(_ + _)

// Collect data for visualization
val (words, counts) = wordCounts.collect().unzip

// Normalize the counts to the range [0, 1] for color mapping
val maxCount = counts.max
val normalizedCounts = counts.map(count => count.toDouble / maxCount)

// Define the HTML code for the bar chart
val html = s"""
  <div id="barchart"></div>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script>
    // Define the data for the bar chart
    var data = [{
      x: [${words.mkString("'", "', '", "'")}],
      y: [${counts.mkString(", ")}],
      type: 'bar',
      marker: {
        color: [${normalizedCounts.mkString(", ")}],
        colorscale: 'Viridis'  // Use any color scale, e.g., 'Viridis'
      }
    }];

    // Define the layout (optional)
    var layout = {
      title: 'Word Frequency',
      xaxis: {title: 'Words'},
      yaxis: {title: 'Frequency'}
    };

    // Create the bar chart
    Plotly.newPlot('barchart', data, layout);
  </script>
"""

// Display the HTML using displayHTML
displayHTML(html)




// Create a text input widget in Databricks
dbutils.widgets.text("userInput", "", "Entrez le texte à prédire ou tapez '-1' pour quitter")

// Retrieve the user's input from the widget
val userInput: String = dbutils.widgets.get("userInput")

// Function to handle negations in the input text
  var isNegation = false
def handleNegations(input: String): Boolean = {
    // Définir une liste de mots de négation
    val negationWords = List("not","don't", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "hardly", "barely", "scarcely", "seldom", "rarely", "without", "unless")

    // Séparer le texte d'entrée en mots
    val words = input.split(" ")

    // Parcourir les mots pour vérifier s'il existe un mot de négation
    for (word <- words) {
        // Si le mot est un mot de négation, retourner true
        if (negationWords.contains(word)) {
            return true
        }
    }

    // Si aucun mot de négation n'est trouvé, retourner false
    return false
}


// Check the user's input
if (userInput != "-1") {
    // Process the user's input with negation handling
    val modifiedInput = handleNegations(userInput)
   

    // Proceed with the rest of your pipeline using the modified input
    val wordsData1 = tokenizer.transform(Seq(userInput).toDF("SentimentText"))
    val swRemovedData1 = swr.transform(wordsData1)
    val featuresData1 = hashTF.transform(swRemovedData1)
    val prediction1 = model.transform(featuresData1)

    // Display the results
  
   val pred_res=prediction1.selectExpr(
    "SentimentText",
  
    s"""CASE 
        WHEN $modifiedInput AND prediction = 0 THEN 1.0
        WHEN $modifiedInput AND prediction = 1 THEN 0.0
        WHEN NOT $modifiedInput AND prediction = 0 THEN 0.0
        WHEN NOT $modifiedInput AND prediction = 1 THEN 1.0
        ELSE NULL
    END AS prediction""",
    // Sélectionnez le type de sentiment (SentimentType) avec une clause CASE
    s"""CASE 
        WHEN $modifiedInput AND prediction = 0 THEN 'positive'
        WHEN $modifiedInput AND prediction = 1 THEN 'negative'
        WHEN NOT $modifiedInput AND prediction = 0 THEN 'negative'
        WHEN NOT $modifiedInput AND prediction = 1 THEN 'positive'
        ELSE 'unknown'
    END AS SentimentType"""
).show(true)
// Import necessary package
import com.databricks.backend.daemon.driver.EnhancedRDDFunctions.displayHTML
// Sélectionnez les colonnes SentimentText, prediction, et SentimentType selon les conditions spécifiées
val predictionResults = prediction1.selectExpr(
    "SentimentText",
    s"""CASE 
        WHEN $modifiedInput AND prediction = 0 THEN 1.0
        WHEN $modifiedInput AND prediction = 1 THEN 0.0
        WHEN NOT $modifiedInput AND prediction = 0 THEN 0.0
        WHEN NOT $modifiedInput AND prediction = 1 THEN 1.0
        ELSE NULL
    END AS prediction""",
    s"""CASE 
        WHEN $modifiedInput AND prediction = 0 THEN 'positive'
        WHEN $modifiedInput AND prediction = 1 THEN 'negative'
        WHEN NOT $modifiedInput AND prediction = 0 THEN 'negative'
        WHEN NOT $modifiedInput AND prediction = 1 THEN 'positive'
        ELSE 'unknown'
    END AS SentimentType"""
)

// Affiche les résultats
// Import necessary packages
import java.math.BigDecimal

// Extraire la première ligne du DataFrame predictionResults
val firstRow = predictionResults.select("prediction").first()

// Extraire la valeur de la colonne prediction
val predictionValue = firstRow.getAs[BigDecimal]("prediction")

// Convertir la valeur de predictionValue (BigDecimal) en Double
val predictionValueAsDouble = predictionValue.doubleValue()




// Determine the color based on the predictionValueAsDouble
val color = if (predictionValueAsDouble == 1) "green" else "red"
val title = if (predictionValueAsDouble == 1) " Positive" else "négative"
// Define the HTML code for the graph
val html = s"""
  <div id="scatterplot"></div>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script>
    var data = [{
      x: [0], // The x-coordinate for the point (can be any value since there is only one point)
      y: [0], // The y-coordinate for the point (can be any value since there is only one point)
      mode: 'markers',
      marker: {
        size: 200, // Size of the circle
        color: '$color' // Color of the circle (green or red)
      }
    }];
    var layout = {
       title: {
        text: 'Prediction de <b style="color:blue;text-decoration: underline overline blue;">$userInput </b> est : $title',
        font: {
          color: '$color' // Set the title color to blue
        }
      },
      xaxis: {showgrid: false, zeroline: false, showticklabels: false},
      yaxis: {showgrid: false, zeroline: false, showticklabels: false},
    };
    Plotly.newPlot('scatterplot', data, layout);
  </script>
"""
 println(s"Vous avez saisi : $userInput")
      prediction1.selectExpr("SentimentText", "prediction","CASE WHEN prediction = 0 THEN 'négative' ELSE 'positive' END AS SentimentType").show(true)

// Display the graph using the HTML code
displayHTML(html)

  

} else {
    println("Vous avez terminé le test")
}


// COMMAND ----------


