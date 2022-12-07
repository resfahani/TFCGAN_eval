import tensorflow as tf
import numpy as np
#import tensorflow.keras.backend as K



def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    reg = tf.nn.relu(fake_loss*fake_loss)

    return fake_loss - real_loss #  + 0.000 * reg


#def discriminator_loss2(real_img, fake_img):
#    return K.mean(real_img * fake_img)

def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


class CWGAN_GP(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,):
        
        super(CWGAN_GP, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.losssave = []
        self.losssave2 = []
        self.custom_loss_mean = []

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(CWGAN_GP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        
    def gradient_penalty(self, batch_size, real_images, fake_images, label ):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([label, interpolated], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    def train_step(self,  data):
        
        real_images, y = data
        
        batch_size = tf.shape(real_images)[0]
        
        #valid = -tf.ones((batch_size, 1))
        #print(valid)
        #fake = tf.ones((batch_size, 1))
        
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                
                # Generate fake images from the latent vector
                fake_images = self.generator([y, random_latent_vectors], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([y, fake_images], training=True)
                # Get the logits for the real images
                real_logits = self.discriminator([y, real_images], training=True)

                # Calculate the discriminator loss using the fake and real image logits
                #d_cost = self.d_loss_fn(real_img=valid, fake_img=real_logits)
                self.d_cost= self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                #d_cost = 0.5 * (d_cost + d_cost2)
                
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images, label = y)
                # Add the gradient penalty to the original discriminator loss
                
                self.d_loss = self.d_cost + gp * self.gp_weight
                self.d2_loss = self.g_loss_fn(real_logits)
                
                #self.custom_loss_mean.append(self.d_loss)
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(self.d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables))

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([y, random_latent_vectors], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([y, generated_images], training=True)
            # Calculate the generator loss
            self.g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(self.g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables))
        
        
        return {"d_loss": self.d_cost, "g_loss": self.g_loss, "d_loss_Fake":self.d2_loss}

    def test_step(self, data):
        # Unpack the data
        real_images, y = data
        batch_size = tf.shape(real_images)[0]
        
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        fake_images = self.generator([y, random_latent_vectors], training=True)
        
        # Get the logits for the fake images
        fake_logits = self.discriminator([y, fake_images], training = True)
        # Get the logits for the real images
        real_logits = self.discriminator([y, real_images], training = True)

        self.Td_cost1= self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)

        self.Tg_loss1 = self.g_loss_fn(real_logits)

        self.Tg_loss2 = self.g_loss_fn(fake_logits)

        return {"d_Validation": self.Td_cost1, "g_Validation_Real": self.Tg_loss1,"g_Validation_Fake": self.Tg_loss2}


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self):
        pass
    def on_train_begin(self, logs=None):
        self.MHistory = []
        #self.MHistoryonBatch = []
        
    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        #self.custom_loss_mean = []
        print(keys)
        
        #print(self.custom_loss_mean)
    def on_train_batch_end(self, batch, logs=None):
        #self.MHistoryonBatch.append(logs)
        pass
    

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print(logs['d_loss'])
        #print(self.custom_loss_mean)
        self.MHistory.append(logs)
        
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
        self.model.generator.save('generator2_EE')
        self.model.discriminator.save('discriminator2_EE')
        
        np.save("./Historyrecord/History_{}".format(epoch), self.MHistory)

        #np.save("HistoryonBatch", self.MHistoryonBatch)
        
        
        
        
