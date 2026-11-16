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


(* Putting types in the place of variables: the compiler's
   [Type_grammar.project_variables_out].

   This file is to be reviewed line by line against the functions
   [project_variables_out] to [project_variant_extensions] of
   middle_end/flambda2/types/grammar/type_grammar.ml, and
   [project_variables_out] of type_descr.ml. It holds definitions only.

   The n-way join of typing environments makes variables, and at its
   end removes those that are mentioned once: a type that is an alias
   of such a variable becomes the type that the variable has, and what
   cannot hold a type forgets the variable
   (types/env/join_env.ml,
   [define_or_eliminate_variables_and_add_equations]).

   What the function does to a type, given the variables to remove
   ([to_project]) and the type to put for each ([expand]):
   - an alias of a variable to remove becomes the type put for the
     variable, which must be of the same kind;
   - a variable for the result of [Is_int], [Get_tag] or [Is_null]
     that is to be removed is replaced by the variable that its type is
     an alias of, if it is an alias of a variable, and is forgotten
     otherwise;
   - an inverse relation on a variable to remove becomes one on the
     name that its type is an alias of, and is forgotten if its type is
     not an alias of a name;
   - an equation of an environment extension on a variable to remove
     is dropped;
   - everything else is traversed.

   The differences:
   - The compiler's function keeps the type that it is given where
     nothing changes (a matter of sharing); the model's builds the type
     again.
   - A type of the model has no coercions, no [Rec_info] expressions
     and no array heads (FlambdaTypeGrammar.v), so
     [project_coercion], [project_head_of_kind_rec_info], the
     [rec_info] of a function type and the cases for arrays are not
     there: an alias of a depth variable that is to be removed becomes
     the type put for it, as for the other kinds.
   - [expand] answers an [option]: [None] is a fatal error of the
     compiler (the type put for a variable is of another kind than the
     alias that it replaces), or what the caller's [expand] answers
     [None] for.
   - A map of the model is a list of which the first binding of a key
     counts; the bindings that do not count are traversed too. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar.
Import ListNotations.

(* [List.map] for a function that may fail, written so that a recursive
   definition over types may go through it. *)
Section Traverse.

  Context {A B : Type} (f : A -> option B).

  Fixpoint traverse (l : list A) : option (list B) :=
    match l with
    | [] => Some []
    | x :: l =>
      match f x with
      | None => None
      | Some y =>
        match traverse l with
        | None => None
        | Some l => Some (y :: l)
        end
      end
    end.

End Traverse.

Section Project.

  (* The variables to remove, and the type to put for each. *)
  Context (to_project : list variable)
    (expand : variable -> option flambda_type).

  (* [project_relation]. *)
  Definition project_relation (relation : option variable)
      : option (option variable) :=
    match relation with
    | None => Some None
    | Some var =>
      if negb (among var to_project)
      then Some relation
      else
        let* ty := expand var in
        Some
          match get_alias_opt ty with
          | Some (Var var') => Some var'
          | Some (Symbol _) | Some (Const _) | None => None
          end
    end.

  (* [project_inverse_relations], for one relation and one name: what
     is left of it. *)
  Definition project_inverse_relation (pair : Relation.t * name)
      : option (list (Relation.t * name)) :=
    match snd pair with
    | Name_symbol _ => Some [pair]
    | Name_var var =>
      if negb (among var to_project)
      then Some [pair]
      else
        let* ty := expand var in
        Some
          match get_alias_opt ty with
          | Some (Var var') => [(fst pair, Name_var var')]
          | Some (Symbol symbol) => [(fst pair, Name_symbol symbol)]
          | Some (Const _) | None => []
          end
    end.

  (* [project_head_of_kind_naked_immediate]. *)
  Definition project_head_of_kind_naked_immediate
      (head : head_of_kind_naked_immediate)
      : option head_of_kind_naked_immediate :=
    let* relations :=
      traverse project_inverse_relation (inverse_relations head)
    in
    Some
      (Head_of_kind_naked_immediate (naked_immediates head)
         (concat relations)).

  (* [Type_descr.project_variables_out]: [of_expanded] reads the type
     put for a variable at the kind of the alias that it replaces. *)
  Definition project_descr {head : Type}
      (project_head : head -> option head)
      (of_expanded : flambda_type -> option (type_descr head))
      (ty : type_descr head) : option (type_descr head) :=
    match ty with
    | Unknown | Bottom => Some ty
    | Ok (No_alias head) =>
      let* head := project_head head in
      Some (Ok (No_alias head))
    | Ok (Equals (Var var)) =>
      if among var to_project
      then
        let* expanded := expand var in
        of_expanded expanded
      else Some ty
    | Ok (Equals (Symbol _)) | Ok (Equals (Const _)) => Some ty
    end.

  (* The cases of [expand_with_coercion] in [project_variables_out]. *)
  Definition value_of_expanded (ty : flambda_type)
      : option (type_descr head_of_kind_value) :=
    match ty with
    | Type_value ty => Some ty
    | Type_naked_immediate _ | Type_naked_number _ _ | Type_rec_info _
    | Type_region _ => None
    end.

  Definition naked_immediate_of_expanded (ty : flambda_type)
      : option (type_descr head_of_kind_naked_immediate) :=
    match ty with
    | Type_naked_immediate ty => Some ty
    | Type_value _ | Type_naked_number _ _ | Type_rec_info _
    | Type_region _ => None
    end.

  Definition naked_number_of_expanded (k : naked_number_kind)
      (ty : flambda_type) : option (type_descr (list Z)) :=
    match ty with
    | Type_naked_number k' ty =>
      if kind_eqb (Naked_number k) (Naked_number k') then Some ty else None
    | Type_value _ | Type_naked_immediate _ | Type_rec_info _
    | Type_region _ => None
    end.

  Definition rec_info_of_expanded (ty : flambda_type)
      : option (type_descr unit) :=
    match ty with
    | Type_rec_info ty => Some ty
    | Type_value _ | Type_naked_immediate _ | Type_naked_number _ _
    | Type_region _ => None
    end.

  Definition region_of_expanded (ty : flambda_type)
      : option (type_descr unit) :=
    match ty with
    | Type_region ty => Some ty
    | Type_value _ | Type_naked_immediate _ | Type_naked_number _ _
    | Type_rec_info _ => None
    end.

  (* A case of a row-like structure of blocks. The two arguments stand
     for the functions being defined below. *)
  Definition project_block_case
      (project_variables_out : flambda_type -> option flambda_type)
      (project_env_extension : env_extension -> option env_extension)
      (case : row_like_block_case) : option row_like_block_case :=
    let* env_extension := project_env_extension (case_env_extension case) in
    let* maps_to := traverse project_variables_out (case_maps_to case) in
    Some (Row_like_case maps_to (case_index case) env_extension).

  (* [project_variables_out] and the functions defined with it. *)
  Fixpoint project_variables_out (t : flambda_type) : option flambda_type :=
    match t with
    | Type_value ty =>
      let* ty :=
        project_descr project_head_of_kind_value value_of_expanded ty
      in
      Some (Type_value ty)
    | Type_naked_immediate ty =>
      let* ty :=
        project_descr project_head_of_kind_naked_immediate
          naked_immediate_of_expanded ty
      in
      Some (Type_naked_immediate ty)
    | Type_naked_number k ty =>
      let* ty :=
        project_descr (fun head => Some head) (naked_number_of_expanded k) ty
      in
      Some (Type_naked_number k ty)
    | Type_rec_info ty =>
      let* ty :=
        project_descr (fun head => Some head) rec_info_of_expanded ty
      in
      Some (Type_rec_info ty)
    | Type_region ty =>
      let* ty := project_descr (fun head => Some head) region_of_expanded ty in
      Some (Type_region ty)
    end

  with project_head_of_kind_value (head : head_of_kind_value)
      : option head_of_kind_value :=
    match head with
    | Head_of_kind_value non_null is_null =>
      let* is_null :=
        match is_null with
        | Not_null => Some Not_null
        | Maybe_null is_null_var =>
          let* is_null_var := project_relation is_null_var in
          Some (Maybe_null is_null_var)
        end
      in
      let* non_null :=
        match non_null with
        | Unknown => Some Unknown
        | Bottom => Some Bottom
        | Ok non_null_head =>
          let* non_null_head :=
            project_head_of_kind_value_non_null non_null_head
          in
          Some (Ok non_null_head)
        end
      in
      Some (Head_of_kind_value non_null is_null)
    end

  with project_head_of_kind_value_non_null
      (head : head_of_kind_value_non_null)
      : option head_of_kind_value_non_null :=
    match head with
    | Variant is_int immediates get_tag blocks extensions is_unique =>
      let* is_int := project_relation is_int in
      let* get_tag := project_relation get_tag in
      let* immediates :=
        match immediates with
        | Or_unknown.Known immediates =>
          let* immediates := project_variables_out immediates in
          Some (Or_unknown.Known immediates)
        | Or_unknown.Unknown => Some Or_unknown.Unknown
        end
      in
      let* blocks :=
        match blocks with
        | Or_unknown.Known blocks =>
          let* blocks := project_row_like_for_blocks blocks in
          Some (Or_unknown.Known blocks)
        | Or_unknown.Unknown => Some Or_unknown.Unknown
        end
      in
      let* extensions :=
        match extensions with
        | No_extensions => Some No_extensions
        | Ext when_immediate when_block =>
          let* when_immediate := project_env_extension when_immediate in
          let* when_block := project_env_extension when_block in
          Some (Ext when_immediate when_block)
        end
      in
      Some (Variant is_int immediates get_tag blocks extensions is_unique)
    | Mutable_block _ | String _ => Some head
    | Boxed_number k contents alloc_mode =>
      let* contents := project_variables_out contents in
      Some (Boxed_number k contents alloc_mode)
    | Closures by_function_slot alloc_mode =>
      let* by_function_slot :=
        project_row_like_for_closures by_function_slot
      in
      Some (Closures by_function_slot alloc_mode)
    end

  with project_row_like_for_blocks (blocks : row_like_for_blocks)
      : option row_like_for_blocks :=
    match blocks with
    | Row_like_for_blocks known_tags other_tags alloc_mode =>
      let* known_tags :=
        traverse
          (fun binding =>
             match snd binding with
             | Or_unknown.Known case =>
               let* case :=
                 project_block_case project_variables_out
                   project_env_extension case
               in
               Some (fst binding, Or_unknown.Known case)
             | Or_unknown.Unknown => Some (fst binding, Or_unknown.Unknown)
             end)
          known_tags
      in
      let* other_tags :=
        match other_tags with
        | Or_bottom.Ok case =>
          let* case :=
            project_block_case project_variables_out project_env_extension
              case
          in
          Some (Or_bottom.Ok case)
        | Or_bottom.Bottom => Some Or_bottom.Bottom
        end
      in
      Some (Row_like_for_blocks known_tags other_tags alloc_mode)
    end

  with project_row_like_for_closures (closures : row_like_for_closures)
      : option row_like_for_closures :=
    match closures with
    | Row_like_for_closures known_closures =>
      let* known_closures :=
        traverse
          (fun binding =>
             let* env_extension :=
               project_env_extension (case_env_extension (snd binding))
             in
             let* maps_to :=
               project_closures_entry (case_maps_to (snd binding))
             in
             Some
               (fst binding,
                Row_like_case maps_to (case_index (snd binding))
                  env_extension))
          known_closures
      in
      Some (Row_like_for_closures known_closures)
    end

  with project_closures_entry (entry : closures_entry)
      : option closures_entry :=
    match entry with
    | Closures_entry function_types closure_types value_slot_types =>
      let* closure_types :=
        traverse
          (fun binding =>
             let* ty := project_variables_out (snd binding) in
             Some (fst binding, ty))
          closure_types
      in
      let* value_slot_types :=
        traverse
          (fun binding =>
             let* ty := project_variables_out (snd binding) in
             Some (fst binding, ty))
          value_slot_types
      in
      Some (Closures_entry function_types closure_types value_slot_types)
    end

  with project_env_extension (extension : env_extension)
      : option env_extension :=
    match extension with
    | Env_extension equations =>
      let* equations :=
        traverse
          (fun binding =>
             let keep_equation :=
               let* ty := project_variables_out (snd binding) in
               Some [(fst binding, ty)]
             in
             match fst binding with
             | Name_symbol _ => keep_equation
             | Name_var var =>
               if among var to_project then Some [] else keep_equation
             end)
          equations
      in
      Some (Env_extension (concat equations))
    end.

  (* [project_variant_extensions]. *)
  Definition project_variant_extensions (extensions : variant_extensions)
      : option variant_extensions :=
    match extensions with
    | No_extensions => Some No_extensions
    | Ext when_immediate when_block =>
      let* when_immediate := project_env_extension when_immediate in
      let* when_block := project_env_extension when_block in
      Some (Ext when_immediate when_block)
    end.

End Project.
