(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Subkinds: whether a value is of what a binder states of it beyond
   its kind.

   This file is to be reviewed, against
   middle_end/flambda2/kinds/flambda_kind.mli
   ([With_subkind.Non_null_value_subkind.t]) and what Simplify reads
   of a subkind into the type of a parameter
   (types/more_type_creators.ml, [unknown_with_subkind]). It holds
   definitions only. The machine uses them: where a handler is applied
   and where the body of code is entered it has no step unless the
   values are of what the parameters state
   (FlambdaSemanticsSpec.apply_cont_value and [call_code], through
   [values_of_subkinds]; the machine with a heap and the machine of
   regions do the same). That is decision D25, at parameters. What
   follows from the test is stated in FlambdaSubkindBinderSpec.v, and
   the order on what two binders state in FlambdaSubkindOrderSpec.v.

   The compiler gives a parameter of a continuation and a parameter of
   code a kind with a subkind ([Bound_parameter.kind], a
   [Flambda_kind.With_subkind.t]): a tagged immediate, a boxed number
   of one kind, a variant with given constant constructors and, for
   each tag, given fields; and whether the value may be null. The type
   of subkinds is in FlambdaKindModel.v, and the metadata of a handler
   and of code state those of their parameters
   (FlambdaModel.hm_params_subkinds, cm_params_subkinds,
   cm_result_subkinds).

   [value_of_subkind] is by recursion on the subkind, to the bottom of
   it. It reads the value, and for a symbol what the static
   environment binds it to; it reads no heap, so a value that is a
   location is of no subkind but [Subkind_anything].

   A variant: a tagged immediate that is one of the constant
   constructors, or a block whose tag is listed. A number that is not
   in the range of a tagged immediate is no constant constructor,
   whatever the list says. For a tag whose fields are given, the block
   has exactly those fields, each of kind [Value] and of its subkind;
   of the entries with one tag the first counts.

   Null and poison. A value that is null is accepted where the binder
   says that the value may be null, and nowhere else: the subkind is
   of a value that is not null. Poison is accepted only where nothing
   is said: where the binder says that the value may be null and gives
   the subkind [Subkind_anything]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaPrimitiveSpec.
Import ListNotations.

Fixpoint value_of_subkind (statics : static_env) (sk : subkind) (v : value)
    {struct sk} : bool :=
  match sk with
  | Subkind_anything => true
  | Subkind_tagged_immediate => is_tagged_immediate v
  | Subkind_boxed_number n =>
    match boxed_number_of statics v with
    | Some c =>
      match boxable_number_of_const c with
      | Some n' => boxable_number_eqb n n'
      | None => false
      end
    | None => false
    end
  | Subkind_variant consts non_consts =>
    match v with
    | Value_const (Tagged_immediate i) =>
      is_tagged_immediate v && existsb (Z.eqb i) consts
    | _ =>
      match block_of statics v with
      | Some (tag, fields) => block_of_shapes statics non_consts tag fields
      | None => false
      end
    end
  end
with block_of_shapes (statics : static_env) (shapes : constructor_shapes)
    (tag : Z) (fields : list value) {struct shapes} : bool :=
  match shapes with
  | Shapes_nil => false
  | Shape_undetermined tag' rest =>
    if Z.eqb tag tag' then true
    else block_of_shapes statics rest tag fields
  | Shape_determined tag' sks rest =>
    if Z.eqb tag tag' then fields_of_subkinds statics sks fields
    else block_of_shapes statics rest tag fields
  end
with fields_of_subkinds (statics : static_env) (sks : field_subkinds)
    (fields : list value) {struct sks} : bool :=
  match sks, fields with
  | Fields_nil, [] => true
  | Fields_cons nullable sk sks, v :: fields =>
    kind_eqb (kind_of_value v) Value
    && match v with
       | Value_const (Poison _) => says_nothing nullable sk
       | Value_const Null => nullable
       | _ => value_of_subkind statics sk v
       end
    && fields_of_subkinds statics sks fields
  | _, _ => false
  end.

(* Whether a value is of what a binder states of it: whether it may be
   null, and its subkind if it is not. *)
Definition value_at (statics : static_env) (ps : param_subkind) (v : value)
    : bool :=
  match v with
  | Value_const (Poison _) => says_nothing (fst ps) (snd ps)
  | Value_const Null => fst ps
  | _ => value_of_subkind statics (snd ps) v
  end.

(* Whether values are of what the binders state of them, one for one.
   Nothing is asked of the values that are left where the list of what
   is stated is the shorter, nor of what is stated beyond the values:
   the empty list asks nothing. The recursion is on what is stated,
   so that the test of the empty list computes with values not yet
   known. *)
Fixpoint values_of_subkinds (statics : static_env) (vs : list value)
    (sks : list param_subkind) {struct sks} : bool :=
  match sks, vs with
  | ps :: sks, v :: vs =>
    value_at statics ps v && values_of_subkinds statics vs sks
  | _, _ => true
  end.
