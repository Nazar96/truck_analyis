ALL_FEATURES = [
    
    'time',
    'timeStop',
    'create_date',
    
    'car_vin', # ТО ЖЕ ЧТО И "CAR_id"? НЕ ВСЕГДА СОВПАДАЮТa
    'imei',
    'license_plate',
    'car_model',
    
    'load_id', # NaN
    'user_id', # NaN
    
    'lat',
    'lon',
    'alt',
    
    'speed',
    'sat', 
    'fuel', # 0 and -1
    'odom', #zeros
    
    'prior', # Zeros
    'stat', # Zeros
    
    
    
     'A1', # mostly zeros
     'A2', # binary
     'A3', # binary
     'A4', # binary
     'D1', # CONST 
     'D2', # CONST
     'D3', # CONST
     'D4', # CONST
    
     'picto', #???
     'inSense', #???
    
     'RS232_0', 
     'RS232_1', 
    
     'is_problem_point', # const
    
    'Acc. Pedal 1 low idle switch',
    'Acc. Pedal Kickdown Switch',
    'Road speed limit status',
    'Acc. Pedal Position (%)',
    'Percent Load at current speed (%)',
    'Remote Acc. Pedal position (%)',# const
    'Engine torque mode',
    "Driver's demand engine - percent torque (%)",
    'Actual engine - percent torque (%)',
    'Engine speed (rpm)',
    'Source address of controlling device for engine control',
    'Engine starter mode',
    'Engine demand, percent torque (%)',
    'Driver 1 working state',
    'Tachograph output shaft speed (rpm)',
    'Tachograph vehicle speed (km/h)',
    'ASR engine control active', # const
    'ASR brake control active', # const
    'Anti-lock braking (ABS) active', # const
    'EBS brake switch',
    'Brake pedal position (%)',
    'ABS off-road switch',
    'ASR off-road switch',
    'ASR "Hill Holder" switch',
    'Traction control override switch',
    'Accelerator interlock switch',
    'Engine derate switch',
    'Auxilary engine shutdown switch',
    'Remote accelerator enable switch',
    'Engine retarder selection',
    'ABS fully operational',
    'EBS red warning signal',
    'ABS/EBS amber warning signal',
    'ATC/ASR information signal',
    'Source address of contolling device for brake control',
    'Trailer ABS status',
    'Tractor mounted trailer ABS warning signal',
    'Trip Distance (km)',
    'Total Vehicle Distance (km)',
    'Trip Fuel (L)',
    'Total Fuel Used (L)',
    'Engine coolant temperature (C)',
    'Fuel temperature (C)',
    'Engine oil temperature 1 (C)',
    'Turbo oil temperature (C)',
    'Engine intercooler temperature (C)',
    'Engine intercooler thermostat opening (%)',
    'Fuel Delivery Pressure (kPa)',
    'Extended Crankcase Blow-by Pressure (kPa)',
    'Engine Oil Level (%)',
    'Engine Oil Pressure (kPa)',
    'Crankcase Pressure (kPa)',
    'Coolant Pressure (kPa)', # CONST
    'Coolant Level (%)',
    'Washer Fluid Level (%)',
    'Fuel Level (%)',
    'Fuel Filter Differential Pressure (kPa)',
    'Engine Oil Filter Differential Pressure (kPa)',
    'Cargo Ambient Temperature (C)',
    'Axle Location',
    'Axle Weight (kg)',
    'Trailer Weight (kg)',
    'Cargo Weight (kg)',
    'Net Battery Current (A)',
    'Alternator Current (A)',
    'Charging System Potential (Voltage) (V)',
    'Battery Potential (V)',
    'Keyswitch Battery Potential (V)',
    'Two Speed Axle Switch',
    'Parking Brake Switch',
    'Cruise Control Pause Switch',
    'Park Brake Release Inhibit Request',
    'Wheel-Based Vehicle Speed (km/h)',
    'Cruise Control Active',
    'Cruise Control Enable Switch',
    'Brake Switch',
    'Clutch Switch',
    'Cruise Control Set Switch',
    'Cruise Control Coast (Decelerate) Switch', #CONST
    'Cruise Control Resume Switch',
    'Cruise Control Accelerate Switch',
    'Cruise Control Set Speed (km/h)',
    'PTO Governor State', # CONST
    'Cruise Control States',
    'Engine Idle Increment Switch',
    'Engine Idle Decrement Switch',
    'Engine Test Mode Switch',
    'Engine Shutdown Override Switch',
    'High Resolution Total Vehicle Distance (km)',
    'High Resolution Trip Distance (km)',
    'High Resolution Engine Trip Fuel (L)',
    'High Resolution Engine Total Fuel Used (L)'
]