package com.dev.neo4j;


import Results.StringResult;
import com.dev.apachepoi.ExtractDataFromExcel;
import org.neo4j.driver.v1.*;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;
import org.yaml.snakeyaml.Yaml;

import java.io.*;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * This class contains run method which accepts gremlin queries as input and queries them on database
 */
public class Neo4jProcedures {
    public static String query = "";
    public static String queryName = "";

    public static String queriesResultFilename = "";
    public static String ResultFilename = "";
    public static String outputFilename = "";
    public static String URI = "";
    public static String username = "";
    public static String password = "";
    public static String outputFilePath = "";
    public static String queriesFilePath = "";
    public static String resultsetFilePath = "";
    public static  String version = "";
	public static  String inputFilePath = "";


    @Procedure(name = "com.Neo4jProcedures.run", mode = Mode.WRITE)
    @Description("CALL com.Neo4jProcedures.run(String said)")
    public Stream<StringResult> run(@Name("said") String said) throws Exception {

        Driver driver = GraphDatabase.driver(URI, AuthTokens.basic(username, password));
        String outputPath = outputFilePath;
        boolean check = false;
        String queryResultPath = outputPath + Neo4jProcedures.queriesResultFilename;
        String resultPath = outputPath + Neo4jProcedures.ResultFilename;
        String asOutputPath =  outputPath + Neo4jProcedures.outputFilename;

        String result = "";
        BufferedWriter writer1 = null;
        BufferedWriter writer2 = null;
        BufferedWriter writer3 = null;

        try(Session s = driver.session() ){
            ExtractDataFromExcel extractDataFromExcel = new ExtractDataFromExcel();
            Map data = extractDataFromExcel.getData(inputFilePath);
            String cypher = data.get(queryName).toString();
            StatementResult cypherOutput = s.writeTransaction(new TransactionWork<StatementResult>() {
                @Override
                public StatementResult execute(Transaction transaction) {
                    StatementResult result = transaction.run(cypher);
                    return result;
                }
            });

            while(cypherOutput.hasNext()){
                Record record = cypherOutput.next();
                System.out.println(record.toString());
            }

        }

        try (Session session = driver.session()) {


            writer1 = new BufferedWriter(new FileWriter(queryResultPath, true));
            writer1.write(query);
            writer1.write("|");
            writer1.write(said);
            writer1.write("|");


            StatementResult statementResult = session.run(said);

            System.out.println("************************");
            List<String> keyList = statementResult.keys();
            keyList.forEach(s -> System.out.println(s));

            int recordCounter = 0;

            while (statementResult.hasNext()) {
                int i = 0;

                if (recordCounter == 0) {
                    Record firstRecord = statementResult.next();
                    System.out.println(firstRecord.toString());

                    Map<String, Object> resultMap = getYamlMap(resultsetFilePath);

                    Object yamlOutput = resultMap.get(query);
                    if (firstRecord.toString().equalsIgnoreCase(yamlOutput.toString())) {
                        System.out.println("Comapring two values ---> equal");
                    }


                    writer1.write(firstRecord.toString());

                    recordCounter++;
                } else {
                    Record record = statementResult.next();

                    Map<String, Object> recordMap = null;

                    System.out.println(record.toString());


                    writer1.write(record.toString());
                    i++;
                }
            }

            writer1.write("|");
                writer1.write("Success");
                writer1.newLine();

            writer2 = new BufferedWriter(new FileWriter(resultPath,true));

            writer2.write(queryName);
            writer2.write("|");
            writer2.write("Success");
            writer2.newLine();




        } catch (Exception e) {
            try {
                writer1.write(" ");
                writer1.write("|");
                writer1.write("Failure");
                writer1.newLine();


                writer2.write(queryName);
                writer2.write("|");
                writer2.write("Failure");
                writer2.newLine();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
            e.printStackTrace();
        } finally {
            writer1.close();
            writer2.close();
        }

        driver.close();
        return Stream.of(new StringResult("Exit"));

    }

    /**
     * This Method return the Map of key value pairs from yaml file
     * @param yamlFilePath
     * @return
     */
    public Map<String, Object> getYamlMap(String yamlFilePath) {
        FileInputStream fis = null;

        try {
            fis = new FileInputStream(new File(yamlFilePath));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        Yaml yaml = new Yaml();
        Map<String, Object> loaded = (Map<String, Object>) yaml.load(fis);
        return loaded;
    }

    /**
     * This method takes parsed yaml data in the form of Map and extract queries from them and call the run() method.
     */
    public void parseYaml() {


        Map<String, Object> loaded = getYamlMap(queriesFilePath);
        for (Map.Entry<String, Object> entry : loaded.entrySet()) {
            System.out.println("Key -->" + entry.getKey() + " : " + "Value -->" + entry.getValue());
            Neo4jProcedures.query = entry.getValue().toString();
            Neo4jProcedures.queryName = entry.getKey().toString();
            String inputQuery = "CALL gremlin.run(\"" + entry.getValue().toString() + "\")";
            System.out.println(inputQuery);


            try {
                Stream<StringResult> streamOutput = new Neo4jProcedures().run(inputQuery);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }

    }

    public static void main(String[] args) {


        Neo4jProcedures neo4jProcedures = new Neo4jProcedures();
        neo4jProcedures.version = args[1].toString();
        queriesResultFilename = "queries_and_status_"+Neo4jProcedures.version+"_"+new Date().getTime() + ".csv";
        ResultFilename = "results_"+Neo4jProcedures.version+"_"+new Date().getTime()+".csv";
        Map<String, Object> propertiesMap = neo4jProcedures.getYamlMap(args[0]);

        neo4jProcedures.URI = propertiesMap.get("URI").toString();
        neo4jProcedures.username = propertiesMap.get("username").toString();
        neo4jProcedures.password = propertiesMap.get("password").toString();
        neo4jProcedures.outputFilePath = propertiesMap.get("outputFilePath").toString();
        neo4jProcedures.queriesFilePath = propertiesMap.get("queriesFilePath").toString();
        neo4jProcedures.resultsetFilePath = propertiesMap.get("resultsetFilePath").toString();
        neo4jProcedures.inputFilePath = propertiesMap.get("inputFilePath").toString();
        neo4jProcedures.parseYaml();


    }
}
