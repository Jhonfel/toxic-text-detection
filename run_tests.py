import unittest

# Descubre y ejecuta todas las pruebas en el directorio 'tests'
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('tests')

# Ejecuta las pruebas
unittest.TextTestRunner().run(test_suite)