import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import base64

# Load model
model = tf.keras.models.load_model("modelcorn.h5")

<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="utf-8">
  <meta content="width=device-width, initial-scale=1.0" name="viewport">

  <title>Cornny</title>
  <meta content="" name="description">
  <meta content="" name="keywords">

  <meta name="theme-color" content="#FFC300">

  <!-- Favicons -->
  <link rel="icon" href="{{url_for('static',filename='img/cornny-icon.svg')}}" type="image/svg+xml">
  <!-- <link href="assets/img/apple-touch-icon.png" rel="apple-touch-icon"> -->

  <!-- Google Fonts -->
  <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,300i,400,400i,600,600i,700,700i|Raleway:300,300i,400,400i,500,500i,600,600i,700,700i|Poppins:300,300i,400,400i,500,500i,600,600i,700,700i" rel="stylesheet">

  <!-- Vendor CSS Files -->
  <link href="{{ url_for('static', filename='vendor/aos/aos.css') }}" rel="stylesheet">
  <link href="{{ url_for('static', filename='vendor/bootstrap/css/bootstrap.min.css') }}" rel="stylesheet">
  <link href="{{ url_for('static', filename='vendor/bootstrap-icons/bootstrap-icons.css') }}" rel="stylesheet">
  <link href="{{ url_for('static', filename='vendor/boxicons/css/boxicons.min.css') }}" rel="stylesheet">
  <link href="{{ url_for('static', filename='vendor/glightbox/css/glightbox.min.css') }}" rel="stylesheet">
  <link href="{{ url_for('static', filename='vendor/remixicon/remixicon.css') }}" rel="stylesheet">
  <link href="{{ url_for('static', filename='vendor/swiper/swiper-bundle.min.css') }}" rel="stylesheet">

  <!-- Template Main CSS File -->
  <link href="{{ url_for('static', filename='css/style.css') }}" rel="stylesheet">

  <!-- Javascript -->
  <script src="{{ url_for('static', filename='js/jquery_3.6.0.min.js') }}" ></script>



</head>

<body>

  <!-- ======= Header ======= -->
  <header id="header" class="fixed-top d-flex align-items-center">
    <div class="container">
      <div class="header-container d-flex align-items-center justify-content-between">
        <div class="logo">
          <h1 class="text-light"><a href="#hero"><span>Cornny</span></a></h1>
          <!-- Uncomment below if you prefer to use an image logo -->
          <!-- <a href="index.html"><img src="assets/img/logo.png" alt="" class="img-fluid"></a> -->
        </div>

        <nav id="navbar" class="navbar">
          <ul>
            <li><a class="nav-link scrollto active" href="#hero">Beranda</a></li>
            <li><a class="nav-link scrollto" href="#about">Tentang</a></li>
            <li><a class="nav-link scrollto" href="#services">Layanan</a></li>
            <!-- <li><a class="nav-link scrollto " href="#portfolio">Portfolio</a></li> -->
            
            <!-- <li class="dropdown"><a href="#"><span>Drop Down</span> <i class="bi bi-chevron-down"></i></a>
              <ul>
                <li><a href="#">Drop Down 1</a></li>
                <li class="dropdown"><a href="#"><span>Deep Drop Down</span> <i class="bi bi-chevron-right"></i></a>
                  <ul>
                    <li><a href="#">Deep Drop Down 1</a></li>
                    <li><a href="#">Deep Drop Down 2</a></li>
                    <li><a href="#">Deep Drop Down 3</a></li>
                    <li><a href="#">Deep Drop Down 4</a></li>
                    <li><a href="#">Deep Drop Down 5</a></li>
                  </ul>
                </li>
                <li><a href="#">Drop Down 2</a></li>
                <li><a href="#">Drop Down 3</a></li>
                <li><a href="#">Drop Down 4</a></li>
              </ul>
            </li>
            <li><a class="nav-link scrollto" href="#contact">Contact</a></li> -->
            <li><a class="getstarted scrollto" href="#about">Get Started</a></li>
          </ul>
          <i class="bi bi-list mobile-nav-toggle"></i>
        </nav><!-- .navbar -->

      </div><!-- End Header Container -->
    </div>
  </header><!-- End Header -->

  <!-- ======= Hero Section ======= -->
  <section id="hero" class="d-flex align-items-center">
    <div class="container text-center position-relative" data-aos="fade-in" data-aos-delay="200">
      <h1>Deteksi Penyakit Jagung Dengan Cornny</h1>
      <h2>Helping the community to keep their corn plants from disease</h2>
      <a href="#about" class="btn-get-started scrollto">Get Started</a>
    </div>
  </section><!-- End Hero -->


    <!-- ======= About Section ======= -->
    <section id="about" class="about">
      <div class="container">

        <div class="row content">
          <div class="col-lg-6" data-aos="fade-right" data-aos-delay="100">
            <h1>Apa itu Cornny?</h1>
            <h3>Kami telah melatih algoritma deep learning untuk mengenali 3 jenis penyakit pada tanaman jagung.</h3>
          </div>
          <div class="col-lg-6 pt-4 pt-lg-0" data-aos="fade-left" data-aos-delay="200">
            <p>
              Cornny merupakan artificial intelligence (Kecerdasan Buatan) yang berbasis website yang dapat mendeteksi penyakit pada tanaman jagung dengan gadget anda sehingga dapat digunakan oleh banyak kalangan dengan mudah. Cornny dilatih untuk mengenali penyakit pada tanaman jagung seperti corn northen leaf blight, grey leaf spot, dan common rust.
            </p>
            <ul>
              <li><i class="ri-check-double-line"></i> Akurasi yang tinggi</li>
              <li><i class="ri-check-double-line"></i> Mudah digunakan</li>
              <li><i class="ri-check-double-line"></i> Gratis</li>
            </ul>
            <p class="fst-italic">
              Cornny dapat membantu anda untuk mengenali penyakit pada tanaman jagung sehingga anda dapat menentukan penanganan yang tepat pada tanaman jagung anda.
            </p>
          </div>
        </div>

      </div>
    </section><!-- End About Section -->

    <!-- ======= Counts Section ======= -->
     <section id="counts" class="counts">
      <div class="container">

        <div class="row counters">

          <div class="col-lg-3 col-6 text-center">
            <span data-purecounter-start="0" data-purecounter-end="574" data-purecounter-duration="1" class="purecounter"></span>
            <p>Data jagung sehat</p>
          </div>

          <div class="col-lg-3 col-6 text-center">
            <span data-purecounter-start="0" data-purecounter-end="1146" data-purecounter-duration="1" class="purecounter"></span>
            <p>Data corn northern leaf blight</p>
          </div>

          <div class="col-lg-3 col-6 text-center">
            <span data-purecounter-start="0" data-purecounter-end="1316" data-purecounter-duration="1" class="purecounter"></span>
            <p>Data grey leaf spot</p>
          </div>

          <div class="col-lg-3 col-6 text-center">
            <span data-purecounter-start="0" data-purecounter-end="1162" data-purecounter-duration="1" class="purecounter"></span>
            <p>Data common rust</p>
          </div>

        </div>

      </div>
    </section> 
    <!-- End Counts Section -->

    <!-- ======= Why Us Section ======= -->
    <section id="why-us" class="why-us">
      <div class="container">

        <div class="row">
          <div class="col-lg-4 d-flex align-items-stretch" data-aos="fade-right">
            <div class="content">
              <h3>Kenapa memilih cornny?</h3>
              <p>
                AI berbasis website untuk deteksi penyakit pada tanaman jagung menggunakan teknologi kecerdasan buatan (AI) dan algoritma deep learning untuk mendiagnosis penyakit pada tanaman jagung melalui gambar yang diunggah ke situs web. Dataset yang digunakan dalam sistem ini terdiri dari ribuan gambar tanaman jagung yang terinfeksi penyakit dan sehat. Saat pengguna mengunggah gambar tanaman jagung, sistem akan menganalisis gambar tersebut dan memberikan diagnosis. Algoritma deep learning digunakan karena dapat mempelajari fitur-fitur kompleks yang terkait dengan penyakit pada tanaman jagung dan menghasilkan diagnosis yang lebih akurat.
              </p>
              <div class="text-center">
                <a href="#services" class="more-btn">Learn More <i class="bx bx-chevron-right"></i></a>
              </div>
            </div>
          </div>
          <div class="col-lg-8 d-flex align-items-stretch">
            <div class="icon-boxes d-flex flex-column justify-content-center">
              <div class="row">
                <div class="col-xl-4 d-flex align-items-stretch" data-aos="zoom-in" data-aos-delay="100">
                  <div class="icon-box mt-4 mt-xl-0">
                    <i class="bx bx-leaf"></i>
                    <h4>common rust</h4>
                    <p>Corn common rust adalah penyakit tanaman jagung yang disebabkan oleh jamur bernama Puccinia sorghi. Penyakit ini sangat umum terjadi pada jagung dan dapat menyebabkan kerusakan pada tanaman, yang pada akhirnya dapat menyebabkan penurunan hasil panen.</p>
                  </div>
                </div>
                <div class="col-xl-4 d-flex align-items-stretch" data-aos="zoom-in" data-aos-delay="200">
                  <div class="icon-box mt-4 mt-xl-0">
                    <i class="bx bx-leaf"></i>
                    <h4>grey leaf spot</h4>
                    <p>Corn grey leaf spot adalah penyakit tanaman jagung yang disebabkan oleh jamur bernama Cercospora zeae-maydis. Penyakit ini dapat menyebabkan kerusakan pada daun jagung dan dapat mengurangi hasil panen pada saat panen.</p>
                  </div>
                </div>
                <div class="col-xl-4 d-flex align-items-stretch" data-aos="zoom-in" data-aos-delay="300">
                  <div class="icon-box mt-4 mt-xl-0">
                    <i class="bx bx-leaf"></i>
                    <h4>Northern leaf blight</h4>
                    <p>Northern leaf blight adalah penyakit tanaman jagung yang disebabkan oleh jamur bernama Exserohilum turcicum (dahulu disebut sebagai Helminthosporium turcicum). Penyakit ini dapat menyebabkan kerusakan pada daun jagung dan dapat mengurangi hasil panen.</p>
                  </div>
                </div>
              </div>
            </div><!-- End .content-->
          </div>
        </div>

      </div>
    </section><!-- End Why Us Section -->

    <!-- ======= Cta Section ======= -->
    <section id="cta" class="cta">
      <div class="container">

        <div class="text-center" data-aos="zoom-in">
          <h3>Call to Action</h3>
          <p> Jaga pertanian jagung Anda dengan teknologi canggih. Temukan deteksi penyakit AI yang cepat dan akurat untuk tanaman Anda. Tingkatkan hasil panen dan perlindungan tanaman Anda.</p>
          <a class="cta-btn" href="#services">Let's Explore</a>
        </div>

      </div>
    </section>
    <!-- End Cta Section -->

    <!-- ======= Services Section ======= -->
    <section id="services" class="services section-bg">
      <div class="container">

        <div class="row">
          <div class="col-lg-4">
            <div class="section-title" data-aos="fade-right">
              <h2>Layanan</h2>
              <p>Kami menyajikan layanan pada Website ini secara Gratis dan <strong>berkualitas!</strong></p>
            </div>
          </div>
          <div class="col-lg-8">
            <div class="row">

              <div class="col-md-6 d-flex align-items-stretch">
                <div class="icon-box" data-aos="zoom-in" data-aos-delay="100">
                  <div class="icon"><i class="bx bx-camera"></i></div>
                    <h4><a>Image Detection</a></h4>
                  <p>Deteksi penyakit pada daun tanaman jagungmu pada gambar daun tanaman jagung yang kamu unggah.</p>
                </div>
              </div>

              
             
              <center>
                
              </center>

            </div>
          </div>
        </div>

      </div>
    </section><!-- End Services Section -->

    <!-- ======= Computer Vision Section ======= -->
    <section id="imagedetection" class="services">
      <div id="aplikasi" class="container">
        <div class="row justify-content-start">
          <div class="col-md-6 left-section">
            <div class="section-title" data-aos="fade-right">
              <h2>Cara Penggunaan Sistem Deteksi</h2>
            </div>
            <div class="step">
              <div class="step-icon">
                <ion-icon name="share-outline"></ion-icon>
              </div>
              <div class="step-content">
                <h4><strong>1. Klik tombol Unggah Gambar</strong></h4>
                <p>Anda perlu menyiapkan gambar daun tanaman jagung anda. Pastikan format gambar yang diunggah adalah .jpg</p>
              </div>
            </div>
            <div class="step">
              <div class="step-icon">
                <ion-icon name="rocket-outline"></ion-icon>
              </div>
              <div class="step-content">
                <h4><strong>2. Menunggu Hasil Keluar</strong></h4>
                <p>Setelah unggah gambar, sistem kami akan memproses dan hasil dari prediksi akan muncul segera.</p>
              </div>
            </div>
            <form>
              <div class="image-preview text-left">
                <label for="input_gambar">Pilih File Gambar</label>
                <input type="file" id="input_gambar" name="file">
              </div>
              <div class="text-center">
                <button id="prediksi_submit" type="submit" class="upload-button">Unggah Gambar</button>
              </div>
            </form>
          </div>
          <div class="col-md-6 right-section text-center">
            <h2><strong>Hasil Prediksi</strong></h2>
            <p>Hasil prediksi akan muncul di sini.</p>
            <div id="hasil_prediksi">

            </div>
          </div>
        </div>
      </div>
    </section>
  

  </main><!-- End #main -->

  

  <!-- ======= Footer ======= -->
  <footer id="footer">  

  <div class="container d-md-flex py-4">

      <div class="me-md-auto text-center text-md-start">
        <div class="copyright">
          &copy; by <strong><span>Cornny </span></strong>. 
        </div>
      </div> 

    </div> 
  </footer>  
<!-- End Footer -->

  <a href="#" class="back-to-top d-flex align-items-center justify-content-center"><i class="bi bi-arrow-up-short"></i></a>


  <!-- ionicon link -->
  <script type="module" src="https://unpkg.com/ionicons@5.5.2/dist/ionicons/ionicons.esm.js"></script>
  <script nomodule src="https://unpkg.com/ionicons@5.5.2/dist/ionicons/ionicons.js"></script>


  <!-- Vendor JS Files -->
  <script src="{{ url_for('static', filename='vendor/purecounter/purecounter_vanilla.js') }}"></script>
  <script src="{{ url_for('static', filename='vendor/aos/aos.js') }}"></script> 
  <script src="{{ url_for('static', filename='vendor/bootstrap/js/bootstrap.bundle.min.js') }}"></script>
  <script src="{{ url_for('static', filename='vendor/glightbox/js/glightbox.min.js') }}"></script>
  <script src="{{ url_for('static', filename='vendor/isotope-layout/isotope.pkgd.min.js') }}"></script>
  <script src="{{ url_for('static', filename='vendor/swiper/swiper-bundle.min.js') }}"></script>
  <script src="{{ url_for('static', filename='vendor/php-email-form/validate.js') }}"></script>

  <!-- Template Main JS File -->
  <script src="{{ url_for('static', filename='js/main.js') }}"></script>
  <script  src="{{ url_for('static', filename='js/client-side.js') }}"></script>
  <!-- <script  src="{{ url_for('static', filename='js/all.js') }}"></script> -->

  <script>
    const msgerForm = get(".msger-inputarea");
    const msgerInput = get(".msger-input");
    const msgerChat = get(".msger-chat");

    // Icons made by Freepik from www.flaticon.com
    const BOT_IMG = "/static/img/cornnybot.png";
    const PERSON_IMG = "/static/img/user.png";
    const BOT_NAME = "CornyBot!";
    const PERSON_NAME = "You";

    msgerForm.addEventListener("submit", (event) => {
      event.preventDefault();

      const msgText = msgerInput.value;
      if (!msgText) return;

      appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
      msgerInput.value = "";
      botResponse(msgText);
    });

    function appendMessage(name, img, side, text) {
      //   Simple solution for small apps
      const msgHTML = `
      <div class="msg ${side}-msg">
        <div class="msg-img" style="background-image: url(${img})"></div>

        <div class="msg-bubble">
          <div class="msg-info">
            <div class="msg-info-name">${name}</div>
            <div class="msg-info-time">${formatDate(new Date())}</div>
          </div>

          <div class="msg-text">${text}</div>
        </div>
      </div>
      `;

      msgerChat.insertAdjacentHTML("beforeend", msgHTML);
      msgerChat.scrollTop += 500;
    }

    function botResponse(rawText) {
      // Bot Response
      $.get("/get", { msg: rawText }).done(function (data) {
        console.log(rawText);
        console.log(data);
        const msgText = data;
        appendMessage(BOT_NAME, BOT_IMG, "left", msgText);
      });
    }

    // Utils
    function get(selector, root = document) {
      return root.querySelector(selector);
    }

    function formatDate(date) {
      const h = "0" + date.getHours();
      const m = "0" + date.getMinutes();

      return `${h.slice(-2)}:${m.slice(-2)}`;
    }
  </script>

</body>

</html>

