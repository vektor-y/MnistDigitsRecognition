package nika.ml.mnist;

import javax.swing.*;
import java.awt.*;

class DigitPanel extends JPanel {

    private final Result result;

    public DigitPanel(Result result) {
        this.result = result;
        setPreferredSize(new Dimension(300, 320));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int size = 28;   // MNIST image size
        int scale = 10;   // enlarge MNIST pixels
        for (int i = 0; i < result.getImage().length; i++) {
            int x = i % size;
            int y = i / size;
            int val = 255 - result.getImage()[i]; // invert so higher means darker
            g.setColor(new Color(val, val, val));
            g.fillRect(x * scale, y * scale, scale, scale);
        }
        g.setColor(Color.BLACK);
        int textY = size * scale + 15;
        g.drawString(String.valueOf("Prediction: " + result.getLabel()), 5, textY);
        g.drawString(String.format("Confidence: %.2f %%", result.getConfidence()), 5, textY + 15);
    }
}