# Base class Vehicle
class Vehicle:
    def move(self):
        raise NotImplementedError("Subclasses must implement this method")

# Derived class Car
class Car(Vehicle):
    def move(self):
        print("Driving")

# Derived class Plane
class Plane(Vehicle):
    def move(self):
        print("Flying")

# Derived class Boat
class Boat(Vehicle):
    def move(self):
        print("Sailing")

# Function that takes any Vehicle and calls its move method
def make_it_move(vehicle: Vehicle):
    vehicle.move()

# Create instances
car = Car()
plane = Plane()
boat = Boat()

# Demonstrate polymorphism
make_it_move(car)    # Output: Driving
make_it_move(plane)  # Output: Flying 
make_it_move(boat)   # Output: Sailing 
