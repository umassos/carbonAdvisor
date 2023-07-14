import json
import urllib.request

countryRegion = {
    'CA-ON' : 'US%20East%20(Ohio)',
    'US-CAL-CISO' : 'US%20West%20(N.%20California)'
}

def get_instance_price(instance_type,country_code):
    region = countryRegion[country_code]
    url = 'https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/' + str(region) +'/Linux/index.json'

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
                return f"Instance price not found"

    except Exception as e:
        return f"An error occurred while fetching the pricing data: {str(e)}"

