package ubc.ecee.cpen502.experience_replay;

import java.util.LinkedList;

public class CircularQueue<E> extends LinkedList<E> {

	private static final long serialVersionUID = 1L;
    private int capacity = 10;

    public CircularQueue(int capacity){
        this.capacity = capacity;
    }

    public boolean add(E e) {
        if(size() >= capacity)
            removeFirst();
        return super.add(e);
    }
}
