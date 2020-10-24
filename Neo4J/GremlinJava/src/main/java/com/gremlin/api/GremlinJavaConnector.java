package com.gremlin.api;

import org.apache.tinkerpop.gremlin.driver.Client;
import org.apache.tinkerpop.gremlin.driver.Cluster;
import org.apache.tinkerpop.gremlin.driver.Result;
import org.apache.tinkerpop.gremlin.driver.ResultSet;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class GremlinJavaConnector {

    public  boolean connect(String query){
        Cluster cluster = null;
        try {
            cluster = Cluster.open("C:\\Users\\GS-2022\\Pictures\\Neo4j Data Learning\\GremlinJava\\src\\main\\resources\\remote.yaml");
        } catch (Exception e) {
            e.printStackTrace();
        }
        Client client = cluster.connect();
        ResultSet result = client.submit(query);
        CompletableFuture<List<Result>> output = result.all();

        try {
            for (Result s : output.get()) {
                System.out.println(s.toString());
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return  true;
    }
    public void run(){

    }
    public static void main(String[] args) {

        String query = "g.V().hasLabel(\\\"Category\\\").values(\\\"categoryName\\\")";
    GremlinJavaConnector gremlinJavaConnector = new GremlinJavaConnector();
    gremlinJavaConnector.connect(query);
    }
}
