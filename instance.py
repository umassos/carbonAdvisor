import json
import urllib.request
import urllib.parse

countryRegion = {
    "CA-ON": "US East (Ohio)",
    "US-CAL-CISO": "US West (N. California)",
    "US-MIDA-PJM": "US East (Ohio)",
    "AU-SA": "Asia Pacific (Sydney)",
    "US-NW-PACW": "US West (Oregon)",
}


def get_instance_price(instance_type, country_code):
    region = countryRegion[country_code]
    parsed_region = urllib.parse.quote(region)
    url = (
        "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/"
        + str(parsed_region)
        + "/Linux/index.json"
    )
    try:
        # Fetch the JSON data from the URL
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

            # Check if the instance type exists in the data

            regions = data["regions"]
            for region in regions:
                instance_types = regions[region]
                for instance in instance_types:
                    if instance_types[instance]["Instance Type"] == instance_type:
                        price = instance_types[instance]["price"]
                        return price
            else:
                return f"Instance price not found"

    except Exception as e:
        return f"An error occurred while fetching the pricing data: {str(e)}"
