package org.apache.hadoop.mypoc;

import org.junit.Test;

import java.io.*;

/**
 * Created by wenjusun on 5/30/16.
 */
public class CheckFileFormatTest {

//    @Test
    public void verifyRecordFormatAPIPartTest()throws IOException{
        String file = "/Users/wenjusun/bigdata/mypoc-hadoop/cloud-service-1.0.log.2016-02-29-23";

        try (BufferedReader br = new BufferedReader(new FileReader(file))){

            String line = br.readLine();

            int i = 0,k=0;
            while(line != null){
//                if (line.contains("[CloudService.Report.API]")) {
                if (line.contains("CloudService#invoke(579)")) {
                    String fields[] = line.split(" ");

                    String apiPart = fields[10];
                    if(apiPart!=null) {
//                        System.out.println(line);
                        i++;

                        String kv[] = apiPart.split("=");
                        if (kv.length == 2) {
                            if (kv[1] == null) {
                                System.out.println(line);
                                i++;
                            }
                        }

                    }
                    else{
                        System.out.println(line);
                        k++;
                        if (k == 20) break;
                    }
                    //if (i == 20) break;
                }

                line = br.readLine();
            }

            System.out.println("========i="+i);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
        }
    }

    //@Test
    public void verifyRecordFormatTest()throws IOException{
        String file = "/Users/wenjusun/bigdata/mypoc-hadoop/cloud-service-1.0.log.2016-02-29-23";

        try (BufferedReader br = new BufferedReader(new FileReader(file))){

            String line = br.readLine();

            int i = 0,k=0;
            while(line != null){
//                if (line.contains("[CloudService.Report.API]")) {
                if (line.contains("CloudService#invoke(579)")) {
                    String fields[] = line.split(" ");

                    String deviceid = fields[13];
                    if(deviceid!=null && deviceid.startsWith("deviceid=")) {
//                        System.out.println(line);
                        i++;
/*
                        String idpair[] = deviceid.split("=");
                        if (idpair.length == 2) {
                            if (idpair[1] == null) {
                                System.out.println(line);
                                i++;
                            }
                        }
*/
                    }
                    else{
                        System.out.println(line);
                        k++;
                        if (k == 20) break;
                    }
                    //if (i == 20) break;
                }

                line = br.readLine();
            }

            System.out.println("========i="+i);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
        }
    }
}
