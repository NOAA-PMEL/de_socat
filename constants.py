from theme import theme

regions = {
    "A": {
        'll': {
            'longitude': -80,
            'latitude': 20
        },
        'ur': {
            'longitude': 0,
            'latitude': 70
        },
        "name": 'North Atlantic'
    },
    "C": {
        'll': {
            'longitude': -180,
            'latitude': -90
        },
        'ur': {
            'longitude': 180,
            'latitude': 90
        }
    },
    "I": {
        'll': {
            'longitude': 20,
            'latitude': -60
        },
        'ur': {
            'longitude': 130,
            'latitude': 30
        }
    },
    "N": {
        'll': {
            'longitude': 110,
            'latitude': 70
        },
        'ur': {
            'longitude': -120,
            'latitude': 0
        }
    },
    "O": {
        'll': {
            'longitude': -180,
            'latitude': -90
        },
        'ur': {
            'longitude': 180,
            'latitude': -60
        }
    },
    "R": {
        'll': {
            'longitude': -180,
            'latitude': 66
        },
        'ur': {
            'longitude': 180,
            'latitude': 90
        }
    },
    "T": {
        'll': {
            'longitude': 120,
            'latitude': -30
        },
        'ur': {
            'longitude': -70,
            'latitude': 30
        }
    },
    "Z": {
        'll': {
            'longitude': -55,
            'latitude': 5
        },
        'ur': {
            'longitude': -15,
            'latitude': 25
        }
    }   
}
region_names = {
    "A" : 'North Atlantic',
    "C" : "Coastal",
    "I" : 'Indian',
    "N" : "North Pacific",
    "O" : "Southern Oceans",
    "R" : "Arctic",
    "T" : "Tropical Pacific",
    "Z" : "Tropical Atlantic"
}


if __name__ == '__main__':
    for id in region_names:
        print(f'{region_names[id]} covers:')
        print(regions[id])         