NAB Data Corpus
---

Data are ordered, timestamped, single-valued metrics. All data files contain anomalies, unless otherwise noted.


### Real data
- realAWSCloudwatch/

	AWS server metrics as collected by the AmazonCloudwatch service. Example metrics include CPU Utilization, Network Bytes In, and Disk Read Bytes.

- realAdExchange/
	
	Online advertisement clicking rates, where the metrics are cost-per-click (CPC) and cost per thousand impressions (CPM). One of the files is normal, without anomalies.
	
- realKnownCause/

	This is data for which we know the anomaly causes; no hand labeling.
	
	- ambient_temperature_system_failure.csv: The ambient temperature in an office setting.
	- cpu_utilization_asg_misconfiguration.csv: From Amazon Web Services (AWS) monitoring CPU usage – i.e. average CPU usage across a given cluster. When usage is high, AWS spins up a new machine, and uses fewer machines when usage is low.
	- ec2_request_latency_system_failure.csv: Data from Amazon's East Coast datacenter, where the anomalies represent AWS API call latencies. There's an interesting story behind this data in the [Numenta blog](http://numenta.com/blog/anomaly-of-the-week.html). 
	- machine_temperature_system_failure.csv: Temperature sensor data of an internal component of a large, industrial mahcine. The first anomaly is a planned shutdown of the machine. The second anomaly is difficult to detect and led to the third anomaly, a catastrophic failure of the machine.
	- nyc_taxi.csv: Number of NYC taxi rides, where the five anomalies occur during the NYC marathon, Thanksgiving, Christmas, New Years day, and a snow storm.
	- rogue_agent_key_hold.csv: Timing the key holds for several users of a computer, where the anomalies represent a change in the user.
	- rogue_agent_key_updown.csv: Timing the key strokes for several users of a computer, where the anomalies represent a change in the user.

- realRogueAgent/

	This data represents computer usage patterns for different users, where an anomaly may occur with a rogue user of the computer.

- realTraffic/

	Traffic data from the Twin Cities Metro area in Minnesota, where the metrics are occupancy, speed, and travel time.

- realTweets/

	A collection of Twitter volume data files from large publicly-traded companies such as Google and IBM.



### Artificial data

- artificialNoAnomaly/

	Artifically-generated data without any anomalies.

- artificialWithAnomaly/

	Artifically-generated data with varying types of anomalies.