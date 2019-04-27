package solver.ls;

import ilog.cp.*;
import ilog.concert.*;

import java.io.File;
import java.io.FileNotFoundException;

import java.util.Scanner;

public class Edge
{
    int startLoc;
    int endLoc;
    double distance;

    public Edge(int startLoc, double startX, double startY,
                int endLoc, double endX, double endY) {

        this.startLoc = startLoc;
        this.endLoc = endLoc;
        this.distance = Math.sqrt(Math.pow(startX - endX, 2) + Math.pow(startY - endY, 2));
    }
}

public class VRPInstance
{
    // VRP Input Parameters
    int numCustomers;               // the number of customers
    int numVehicles;            // the number of vehicles
    int vehicleCapacity;            // the capacity of the vehicles
    int[] demandOfCustomer;     // the demand of each customer
    double[] xCoordOfCustomer;  // the x coordinate of each customer
    double[] yCoordOfCustomer;  // the y coordinate of each customer

    // Things we added
    IloCP cp;

    public VRPInstance(String fileName) {
        Scanner read = null;
        try {
            read = new Scanner(new File(fileName));
        } catch (FileNotFoundException e) {
            System.out.println("Error: in VRPInstance() " + fileName + "\n" + e.getMessage());
            System.exit(-1);
        }

        numCustomers = read.nextInt();
        numVehicles = read.nextInt();
        vehicleCapacity = read.nextInt();

        System.out.println("Number of customers: " + numCustomers);
        System.out.println("Number of vehicles: " + numVehicles);
        System.out.println("Vehicle capacity: " + vehicleCapacity);

        demandOfCustomer = new int[numCustomers];
        xCoordOfCustomer = new double[numCustomers];
        yCoordOfCustomer = new double[numCustomers];

        for (int i = 0; i < numCustomers; i++) {
            demandOfCustomer[i] = read.nextInt();
            xCoordOfCustomer[i] = read.nextDouble();
            yCoordOfCustomer[i] = read.nextDouble();
        }

        for (int i = 0; i < numCustomers; i++) {
            System.out.println(demandOfCustomer[i] + " " + xCoordOfCustomer[i] + " " + yCoordOfCustomer[i]);
        }
    }

    public void solve() {
        cp = new IloCP();

        // Edges (A <-> B)
        //
        // Locations (A)
        //      if (A <-> B) in set of truck's edges => A's demand adds to truck usage
        int numEdges = Math.ceil(numCustomers * (numCustomers - 1));
        for (int )





        IloIntVar[][] truckE = cp.intVarArray(numEdges, 0, numTrucks);


        // Constraint 1: The sum of all demands met by a truck have to be less
        // than or equal to the capacity of the truck.
        for (int i = 0; i < numEdges; i++) {
            truckE[i]
        }




        // Constraint 2: Exactly 1 truck visits each customer.


        // Objective: Minimize the total distance travelled by all trucks.






        // // Truck Sequences {T_1, T_2, ..., T_n} where n = number of locations
        // //
        // //
        // //
        // cp.


        // // Variables:
        // IloIntVar[][] seqT = new IloIntVar[numVehicles][numCustomers];
        // for (int i = 0; i < numVehicles; i++) {
        //     seqT[i] = cp.intVarArray(numCustomers, -numCustomers, numCustomers); // -1 === "unused"
        // }

        // cp.allDiff(seqT[i]);
    }
}
