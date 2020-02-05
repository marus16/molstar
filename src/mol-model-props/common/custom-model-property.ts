/**
 * Copyright (c) 2019-2020 mol* contributors, licensed under MIT, See LICENSE file for more info.
 *
 * @author Alexander Rose <alexander.rose@weirdbyte.de>
 */

import { CustomPropertyDescriptor, Model } from '../../mol-model/structure';
import { ParamDefinition as PD } from '../../mol-util/param-definition';
import { ValueBox } from '../../mol-util';
import { CustomProperty } from './custom-property';

export { CustomModelProperty }

namespace CustomModelProperty {
    export interface Provider<Params extends PD.Params, Value> extends CustomProperty.Provider<Model, Params, Value> { }

    export interface ProviderBuilder<Params extends PD.Params, Value> {
        readonly label: string
        readonly descriptor: CustomPropertyDescriptor
        readonly defaultParams: Params
        readonly getParams: (data: Model) => Params
        readonly isApplicable: (data: Model) => boolean
        readonly obtain: (ctx: CustomProperty.Context, data: Model, props: PD.Values<Params>) => Promise<Value>
        readonly type: 'static' | 'dynamic'
    }

    export function createProvider<Params extends PD.Params, Value>(builder: ProviderBuilder<Params, Value>): CustomProperty.Provider<Model, Params, Value> {
        const descriptorName = builder.descriptor.name
        const propertyDataName = builder.type === 'static' ? '_staticPropertyData' : '_dynamicPropertyData'

        const get = (data: Model) => {
            if (!(descriptorName in data[propertyDataName])) {
                (data[propertyDataName][descriptorName] as CustomProperty.Container<PD.Values<Params>, Value>) = {
                    props: { ...PD.getDefaultValues(builder.getParams(data)) },
                    data: ValueBox.create(undefined)
                }
            }
            return data[propertyDataName][descriptorName] as CustomProperty.Container<PD.Values<Params>, Value>;
        }
        const set = (data: Model, props: PD.Values<Params>, value: Value | undefined) => {
            const property = get(data);
            (data[propertyDataName][descriptorName] as CustomProperty.Container<PD.Values<Params>, Value>) = {
                props,
                data: ValueBox.withValue(property.data, value)
            };
        }

        return {
            label: builder.label,
            descriptor: builder.descriptor,
            getParams: builder.getParams,
            isApplicable: builder.isApplicable,
            attach: async (ctx: CustomProperty.Context, data: Model, props: Partial<PD.Values<Params>> = {}) => {
                const property = get(data)
                const p = { ...property.props, ...props }
                if (property.data.value && PD.areEqual(builder.defaultParams, property.props, p)) return
                const value = await builder.obtain(ctx, data, p)
                data.customProperties.add(builder.descriptor);
                set(data, p, value);
            },
            get: (data: Model) => get(data)?.data,
            set: (data: Model, props: Partial<PD.Values<Params>> = {}) => {
                const property = get(data)
                const p = { ...property.props, ...props }
                if (!PD.areEqual(builder.defaultParams, property.props, p)) {
                    // this invalidates property.value
                    set(data, p, undefined)
                }
            }
        }
    }
}