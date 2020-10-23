# Neo4j Plugin Development

This project contains java code to run gremlin queries through gremlin.run() procedure.

## Prerequisites

Java 1.8
Neo4j Enterprise Edition 3.2.3/3.5.8
Maven


## Build Jars

Use the maven to build jars.
Go to the base directory where pom.xml resides and hit the following command

```bash
mvn package
```
After that, jars will be created in the target folder.

## Usage
Files Needed to run the program :
1. neo4j-createProcedure.jar
2. queries.yml
3. resultset.yml
4. properties.yml

```bash
java -cp <path to jar> com.dev.neo4j.Neo4jProcedures <path to properties.yml>
```
Make necessary changes in properties.yml file according to platform(Windows/Linux) on which running java jar.

## Authors
Kunal Patil
