/**
 * Copyright (c) 2017 mol* contributors, licensed under MIT, See LICENSE file for more info.
 *
 * @author David Sehnal <david.sehnal@gmail.com>
 */

import Writer from './writer'

interface Encoder<T> {
    encode(): void,
    writeTo(writer: Writer<T>): void
}

export default Encoder