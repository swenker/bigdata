package org.apache.hadoop.mypoc;


import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

/**
 * Created by wenjusun on 5/26/16.
 */
public class LogReducer extends Reducer<Text,IntWritable,Text,IntWritable> {

    IntWritable resultSum= new IntWritable();

    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {

        int sum = 0;
        for(IntWritable s:values){
            sum += s.get();
        }

        resultSum.set(sum);
        context.write(key,resultSum);
    }
}
