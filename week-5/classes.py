# Base class representing a generic ElectronicDevice
class ElectronicDevice:
    def __init__(self, brand, model, power_on=False):
        self.brand = brand
        self.model = model
        self.power_on = power_on  # encapsulated attribute to track power state

    def power_on_device(self):
        self.power_on = True
        print(f"{self.brand} {self.model} is now ON.")

    def power_off_device(self):
        self.power_on = False
        print(f"{self.brand} {self.model} is now OFF.")

    def device_info(self):
        return f"Brand: {self.brand}, Model: {self.model}, Power: {'ON' if self.power_on else 'OFF'}"


# Derived class representing a Smartphone that inherits from ElectronicDevice
class Smartphone(ElectronicDevice):
    def __init__(self, brand, model, os, storage_gb, power_on=False):
        super().__init__(brand, model, power_on)  # call parent constructor
        self.os = os
        self.storage_gb = storage_gb

    # Method overriding to include smartphone-specific info
    def device_info(self):
        base_info = super().device_info()
        return f"{base_info}, OS: {self.os}, Storage: {self.storage_gb}GB"

    def install_app(self, app_name):
        if self.power_on:
            print(f"Installing {app_name} on {self.brand} {self.model}...")
        else:
            print(f"Turn on the device before installing apps.")


# Example usage
phone = Smartphone("Samsung", "Galaxy S21", "Android", 128)
print(phone.device_info())  # prints initial info with power OFF
phone.power_on_device()
phone.install_app("WhatsApp")
print(phone.device_info())  # prints info with power ON
