/**
 * Copyright (c) 2018 mol* contributors, licensed under MIT, See LICENSE file for more info.
 *
 * @author Alexander Rose <alexander.rose@weirdbyte.de>
 */

import { Unit } from 'mol-model/structure';
import { RepresentationProps, Visual } from '..';
import { DefaultStructureMeshProps } from '.';
import { RuntimeContext } from 'mol-task';
import { PickingId } from '../../util/picking';
import { LocationIterator } from './visual/util/location-iterator';
import { Mesh } from '../../shape/mesh';
import { MarkerAction, applyMarkerAction } from '../../util/marker-data';
import { Loci, isEveryLoci } from 'mol-model/loci';
import { MeshRenderObject } from 'mol-gl/render-object';
import { createUnitsMeshRenderObject, createColors } from './visual/util/common';
import { deepEqual, ValueCell } from 'mol-util';
import { updateMeshValues, updateRenderableState } from '../util';
import { Interval } from 'mol-data/int';

export interface UnitsVisual<P extends RepresentationProps = {}> extends Visual<Unit.SymmetryGroup, P> { }

export const DefaultUnitsMeshProps = {
    ...DefaultStructureMeshProps,
    unitKinds: [ Unit.Kind.Atomic, Unit.Kind.Spheres ] as Unit.Kind[]
}
export type UnitsMeshProps = typeof DefaultUnitsMeshProps

export interface UnitsMeshVisualBuilder<P extends UnitsMeshProps> {
    defaultProps: P
    createMesh(ctx: RuntimeContext, unit: Unit, props: P, mesh?: Mesh): Promise<Mesh>
    createLocationIterator(group: Unit.SymmetryGroup): LocationIterator
    getLoci(pickingId: PickingId, group: Unit.SymmetryGroup, id: number): Loci
    mark(loci: Loci, group: Unit.SymmetryGroup, apply: (interval: Interval) => boolean): boolean
}

export function UnitsMeshVisual<P extends UnitsMeshProps>(builder: UnitsMeshVisualBuilder<P>): UnitsVisual<P> {
    const { defaultProps, createMesh, createLocationIterator, getLoci, mark } = builder

    let renderObject: MeshRenderObject
    let currentProps: P
    let mesh: Mesh
    let currentGroup: Unit.SymmetryGroup
    let locationIt: LocationIterator

    return {
        get renderObject () { return renderObject },
        async create(ctx: RuntimeContext, group: Unit.SymmetryGroup, props: Partial<P> = {}) {
            currentProps = Object.assign({}, defaultProps, props)
            currentGroup = group

            const unit = group.units[0]
            mesh = currentProps.unitKinds.includes(unit.kind)
                ? await createMesh(ctx, unit, currentProps, mesh)
                : Mesh.createEmpty(mesh)

            locationIt = createLocationIterator(group)
            renderObject = createUnitsMeshRenderObject(group, mesh, locationIt, currentProps)
        },
        async update(ctx: RuntimeContext, props: Partial<P>) {
            const newProps = Object.assign({}, currentProps, props)

            if (!renderObject) return false

            let updateColor = false

            // TODO create in-place
            // if (currentProps.radialSegments !== newProps.radialSegments) return false

            // TODO
            // if (newProps.detail !== currentProps.detail) {
            //     const unit = currentGroup.units[0]
            //     const radius = getElementRadius(unit, newProps.sizeTheme)
            //     mesh = await createElementSphereMesh(ctx, unit, radius, newProps.detail, mesh)
            //     ValueCell.update(renderObject.values.drawCount, mesh.triangleCount * 3)
            //     updateColor = true
            // }

            if (!deepEqual(newProps.colorTheme, currentProps.colorTheme)) {
                updateColor = true
            }

            if (updateColor) {
                createColors(createLocationIterator(currentGroup), newProps.colorTheme, renderObject.values)
            }

            updateMeshValues(renderObject.values, newProps)
            updateRenderableState(renderObject.state, newProps)

            currentProps = newProps
            return true
        },
        getLoci(pickingId: PickingId) {
            return getLoci(pickingId, currentGroup, renderObject.id)
        },
        mark(loci: Loci, action: MarkerAction) {
            const { tMarker } = renderObject.values
            const { elementCount, instanceCount } = locationIt

            function apply(interval: Interval) {
                const start = Interval.start(interval)
                const end = Interval.end(interval)
                return applyMarkerAction(tMarker.ref.value.array, start, end, action)
            }

            let changed = false
            if (isEveryLoci(loci)) {
                apply(Interval.ofBounds(0, elementCount * instanceCount))
                changed = true
            } else {
                changed = mark(loci, currentGroup, apply)
            }
            if (changed) {
                ValueCell.update(tMarker, tMarker.ref.value)
            }
        },
        destroy() {
            // TODO
        }
    }
}