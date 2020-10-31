import java.util.Properties;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.api.java.operators.AggregateOperator;
import org.apache.flink.api.java.operators.ReduceOperator;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.FileSystem.WriteMode;
import org.apache.flink.graph.Edge;
import org.apache.flink.graph.Graph;
import org.apache.flink.graph.Vertex;
import org.apache.flink.graph.library.SingleSourceShortestPaths;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.JsonNode;
import org.apache.flink.shaded.jackson2.com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.assigners.TumblingProcessingTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.streaming.connectors.twitter.TwitterSource;
import org.apache.flink.types.NullValue;
import org.apache.flink.util.Collector;

public class StreamProcessing {
	
	public  static class Tweet{
		
		private String language;
		private String text;
		public Tweet(String language, String text) {
			super();
			this.language = language;
			this.text = text;
		}
		public String getLanguage() {
			return language;
		}
		public void setLanguage(String language) {
			this.language = language;
		}
		public String getText() {
			return text;
		}
		public void setText(String text) {
			this.text = text;
		}
		@Override
		public String toString() {
			return "Tweet [language=" + language + ", text=" + text + "]";
		}
			
	}
	
	public  static class FlatMapper implements FlatMapFunction<String, Tweet> {
		/**
		 * 
		 */
		private static final long serialVersionUID = -427421119126356977L;
		ObjectMapper mapper = new ObjectMapper();
		

	
		public void flatMap(String value, Collector<Tweet> out) throws Exception {
			// TODO Auto-generated method stub
			JsonNode tweet = mapper.readTree(value);
			JsonNode textNode  ;
			JsonNode langNode;
		
			if(tweet.has("text") && tweet.has("lang")) 
			{
			 textNode = tweet.get("text");
			 langNode = tweet.get("lang");
			
			String text = "";
			if( textNode.textValue() != null) {
				text= textNode.asText() ;
			}
			String lang = "";
			if( langNode.textValue()!=null) {
				lang = langNode.asText();
			}
			out.collect(new Tweet(lang, text));
			}
			
		}

		
		}
		

		public static void main(String[] args) throws Exception {
			
			StreamExecutionEnvironment environment = StreamExecutionEnvironment.getExecutionEnvironment();
			Properties properties = new Properties();
			
			properties.setProperty(TwitterSource.CONSUMER_KEY, args[0]);
			properties.setProperty(TwitterSource.CONSUMER_SECRET, args[1]);
			properties.setProperty(TwitterSource.TOKEN, args[2]);
			properties.setProperty(TwitterSource.TOKEN_SECRET, args[3]);
			environment.setParallelism(1);
			environment.addSource(new TwitterSource(properties))
		.flatMap(new FlatMapper())
		.keyBy(new KeySelector<StreamProcessing.Tweet, String>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 11327562463591728L;

			public String getKey(Tweet value) throws Exception {
				// TODO Auto-generated method stub
				return value.getLanguage();
			}
		})
		.window(TumblingProcessingTimeWindows.of(Time.seconds(10)))
		.apply(new WindowFunction<StreamProcessing.Tweet, Tuple2<String, Integer>, String, TimeWindow>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = -2537048017171459218L;
			
			public void apply(String key, TimeWindow window, Iterable<Tweet> input, Collector<Tuple2<String, Integer>> out) throws Exception {
				int count=0;
				for(Tweet tweet : input) {
					count++;
				}
				System.out.println("-------->");
				out.collect(new Tuple2<String, Integer>(key, count));
			}
		})
		
		.print();
			
			environment.execute();
		}


	
		
}




