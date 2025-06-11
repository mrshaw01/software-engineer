class ElectricVehicle:

    def __init__(self, battery_capacity, charging_time):
        self.battery_capacity = battery_capacity
        self.charging_time = charging_time

    def charge_battery(self):
        print("Charging battery...")


class GasolineVehicle:

    def __init__(self, fuel_tank_capacity, fuel_efficiency):
        self.fuel_tank_capacity = fuel_tank_capacity
        self.fuel_efficiency = fuel_efficiency

    def refuel(self):
        print("Refueling...")


class HybridCar(ElectricVehicle, GasolineVehicle):

    def __init__(self, battery_capacity, charging_time, fuel_tank_capacity, fuel_efficiency):
        ElectricVehicle.__init__(self, battery_capacity, charging_time)
        GasolineVehicle.__init__(self, fuel_tank_capacity, fuel_efficiency)

    def drive(self):
        print("Driving with electric and gasoline power.")


car = HybridCar(60, 4, 40, 30)
car.charge_battery()
car.refuel()
car.drive()
