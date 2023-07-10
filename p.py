import pandas as pd
df = pd.read_csv("./csvs/Aalto_test_IoTDevID.csv")

f = open("Aalto_Test_Frequencies.txt", "w")

features= ['pck_size', 'Ether_type', 'LLC_ctrl','EAPOL_version', 'EAPOL_type', 'IP_ihl','IP_tos', 'IP_len', 'IP_flags', 'IP_DF','IP_ttl', 'IP_options', 'ICMP_code', 'TCP_dataofs','TCP_FIN', 'TCP_ACK', 'TCP_window', 'UDP_len','DHCP_options', 'BOOTP_hlen', 'BOOTP_flags', 'BOOTP_sname','BOOTP_file', 'BOOTP_options', 'DNS_qr', 'DNS_rd', 'DNS_qdcount', 'dport_class', 'payload_bytes', 'entropy']

for i,j in enumerate(features):
	f.write("Num : " + str(i) + " ----  " + str(j) + "\n")
	f.write(str(df[j].value_counts()))
	f.write("\nMax : " + str(df[j].max()))
	f.write("\nMin : " + str(df[j].min()))
	f.write("\nMean : " + str(df[j].mean()))
	f.write("\n" * 6)