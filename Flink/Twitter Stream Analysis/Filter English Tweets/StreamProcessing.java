import java.util.Properties;

import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
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
			
			properties.setProperty(TwitterSource.CONSUMER_KEY, "OhbUbxisFS4J6quuKfQrQ1tyN");
			properties.setProperty(TwitterSource.CONSUMER_SECRET, "LpcsvemnOamvUr1JgaQFxpfGAmSawcwCDEgfJ9ms6TzjEG8BcD");
			properties.setProperty(TwitterSource.TOKEN, "1100640992695279618-SHAXnPQZnmWucrCe3bDZCxuk1nAXZ5");
			properties.setProperty(TwitterSource.TOKEN_SECRET, "Xt0K4m8077v2ZDhTt5qt0RH9aYI1Ulunfm8UHcqKJxUoa");
			
			environment.addSource(new TwitterSource(properties))
		.flatMap(new FlatMapper())
		.filter(new FilterFunction<StreamProcessing.Tweet>() {
			
			/**
			 * 
			 */
			private static final long serialVersionUID = -1171526676155102413L;

			public boolean filter(Tweet value) throws Exception {
				// TODO Auto-generated method stub
				
				
				return value.getLanguage().equalsIgnoreCase("en");
			}
		})
		.print();
			
			environment.execute();
		}


	
		
}


