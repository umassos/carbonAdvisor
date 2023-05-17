import json
import urllib.request

def get_instance_price(instance_type):
    url = 'https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(Ohio)/Linux/index.json'

    try:
        # Fetch the JSON data from the URL
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

            # Check if the instance type exists in the data

            regions = data['regions']
            for region in regions:
                instance_types = regions[region]
                for instance in instance_types:
                    if instance_types[instance]['Instance Type'] == instance_type:
                        price = instance_types[instance]['price']
                        return price
            else:
                return -1

    except Exception as e:
        return f"An error occurred while fetching the pricing data: {str(e)}"

