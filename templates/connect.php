<?php
// Database configuration
$servername = "localhost";
$username = "root";
$password = "";
$database = "dual_stroke"; // Make sure this matches your created DB

// Create connection
$conn = new mysqli($servername, $username, $password, $database);

// Check database connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Collect data from form
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = htmlspecialchars($_POST["name"]);
    $email = htmlspecialchars($_POST["email"]);
    $password_plain = $_POST["password"];

    // Hash password for security
    $hashed_password = password_hash($password_plain, PASSWORD_DEFAULT);

    // Insert into the users table
    $stmt = $conn->prepare("INSERT INTO users (name, email, password) VALUES (?, ?, ?)");
    $stmt->bind_param("sss", $name, $email, $hashed_password);

    if ($stmt->execute()) {
        // Redirect to success page
        header("Location: success.html");
        exit();
    } else {
        echo "Error: " . $stmt->error;
    }

    // Close statement and connection
    $stmt->close();
    $conn->close();
}
?>
