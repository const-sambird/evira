import sys
from qiskit_ibm_runtime import QiskitRuntimeService

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python', sys.argv[0], '[QISKIT_API_KEY]')
        exit(1)
    QiskitRuntimeService.save_account(
        token=sys.argv[1]
    )
