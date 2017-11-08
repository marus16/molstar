/**
 * Copyright (c) 2017 mol* contributors, licensed under MIT, See LICENSE file for more info.
 *
 * @author David Sehnal <david.sehnal@gmail.com>
 */

import AtomSet from '../set'
import Atom from '../../atom'
import { OrderedSet } from 'mol-data/int'
import { sortArray } from 'mol-data/util/sort'

export class Builder {
    private keys: number[] = [];
    private units: number[][] = Object.create(null);
    private currentUnit: number[] = [];

    atomCount = 0;

    add(u: number, a: number) {
        const unit = this.units[u];
        if (!!unit) { unit[unit.length] = a; }
        else {
            this.units[u] = [a];
            this.keys[this.keys.length] = u;
        }
        this.atomCount++;
    }

    beginUnit() { this.currentUnit = this.currentUnit.length > 0 ? [] : this.currentUnit; }
    addToUnit(a: number) { this.currentUnit[this.currentUnit.length] = a; this.atomCount++; }
    commitUnit(u: number) {
        if (this.currentUnit.length === 0) return;
        this.keys[this.keys.length] = u;
        this.units[u] = this.currentUnit;
    }

    getSet(): AtomSet {
        const sets: { [key: number]: OrderedSet } = Object.create(null);

        let allEqual = this.keys.length === AtomSet.unitCount(this.parent);

        for (let i = 0, _i = this.keys.length; i < _i; i++) {
            const k = this.keys[i];
            const unit = this.units[k];
            const l = unit.length;
            if (!this.sorted && l > 1) sortArray(unit);

            const set = l === 1 ? OrderedSet.ofSingleton(unit[0]) : OrderedSet.ofSortedArray(unit);
            const parentSet = AtomSet.unitGetById(this.parent, k);
            if (OrderedSet.areEqual(set, parentSet)) {
                sets[k] = parentSet;
            } else {
                sets[k] = set;
                allEqual = false;
            }
        }
        return allEqual ? this.parent : AtomSet.create(sets);
    }

    singleton(): Atom {
        const u = this.keys[0];
        return Atom.create(u, this.units[u][0]);
    }

    constructor(private parent: AtomSet, private sorted: boolean) { }
}

export default function createBuilder(parent: AtomSet, sorted: boolean) {
    return new Builder(parent, sorted);
}