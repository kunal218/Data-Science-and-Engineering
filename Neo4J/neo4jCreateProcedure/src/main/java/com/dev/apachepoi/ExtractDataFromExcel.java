package com.dev.apachepoi;

import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class ExtractDataFromExcel  {

    public Map getData(String filePath){

        try {
            File myFile = new File(filePath);
            FileInputStream fis = new FileInputStream(myFile);

            Map <String,String> data = new HashMap();
            //HashedMap<String,String> data = new HashedMap<>();
            // Finds the workbook instance for XLSX file
            XSSFWorkbook myWorkBook = new XSSFWorkbook (fis);

            // Return first sheet from the XLSX workbook
            XSSFSheet mySheet = myWorkBook.getSheetAt(0);

            // Get iterator to all the rows in current sheet
            Iterator<Row> rowIterator = mySheet.iterator();
            Row firstRow = rowIterator.next();
            // Traversing over each row of XLSX file
            while (rowIterator.hasNext()) {

                Row row = rowIterator.next();

                data.putIfAbsent(row.getCell(0).toString(),row.getCell(2).toString());


            }


            /*for(  Map.Entry<String, String> entry : data.entrySet()){
                System.out.println("Key  :" +entry.getKey()+" value :"+entry.getValue());
                System.out.println("*************************************");

            }*/
           return data;

        }catch (IOException e){
            e.printStackTrace();
        }

        return null;
    }
    public static void main(String[] args) {
        ExtractDataFromExcel object = new ExtractDataFromExcel();
       // object.getData("C:\\Users\\GS-2022\\Pictures\\tinkerpopfull\\neo4jCreateProcedure\\src\\main\\resources\\cypher_queris_northwind.xlsx");

    }

}
